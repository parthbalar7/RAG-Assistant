"""
scripts/migrate_embeddings.py — Re-embed every ChromaDB collection with the configured model.

Required after changing RAG_EMBEDDING_MODEL (e.g. all-MiniLM-L6-v2 -> Alibaba-NLP/gte-modernbert-base):
stored vectors keep the old model's dimension and VectorStore refuses to start on the mismatch.

Iterates ALL collections in the persist dir (docs_*, memories_*, tech_docs, ...), re-embeds
documents in batches of 64 via core.retriever.embed_texts, and writes each collection back
through a temp-collection swap. The original is parked as {name}__migrate_backup until the
new collection holds the canonical name, so a mid-run crash never loses data; leftovers from
a crashed run are detected and restored (never silently deleted) at the next start.

Usage:
    .venv\\Scripts\\python.exe scripts/migrate_embeddings.py [--model MODEL_ID] [--dry-run]
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BATCH_SIZE = 64
TMP_SUFFIX = "__migrate_tmp"
BACKUP_SUFFIX = "__migrate_backup"


def _collection_names(client) -> list[str]:
    # Chroma >= 0.6 returns names (str); older versions return Collection objects
    return sorted(c if isinstance(c, str) else c.name for c in client.list_collections())


def _stored_dimension(collection) -> int | None:
    try:
        sample = collection.get(limit=1, include=["embeddings"])
        embeddings = sample.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return len(embeddings[0])
    except Exception:
        return None


def _migrate_collection(client, name: str, embed_texts) -> int:
    """Re-embed *name* into a temp collection, then atomically swap names. Returns docs migrated."""
    source = client.get_collection(name)
    count = source.count()
    if count == 0:
        print(f"  {name}: empty, skipped")
        return 0

    tmp_name = f"{name}{TMP_SUFFIX}"
    with contextlib.suppress(Exception):  # leftover from a previous failed run
        client.delete_collection(tmp_name)
    tmp = client.create_collection(name=tmp_name, metadata=source.metadata or {"hnsw:space": "cosine"})

    done = 0
    offset = 0
    while True:
        page = source.get(limit=BATCH_SIZE, offset=offset, include=["documents", "metadatas"])
        ids = page["ids"]
        if not ids:
            break
        documents = [doc or "" for doc in page["documents"]]
        metadatas = page["metadatas"]
        embeddings = embed_texts(documents, persist=False)
        tmp.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        offset += len(ids)
        done += len(ids)
        print(f"  {name}: {done}/{count} re-embedded")

    # Crash-safe swap: the intact original survives (as the backup) until the
    # re-embedded collection holds the canonical name; _recover_leftovers
    # repairs any interruption on the next run.
    backup_name = f"{name}{BACKUP_SUFFIX}"
    source.modify(name=backup_name)
    tmp.modify(name=name)
    client.delete_collection(backup_name)
    return done


def _recover_leftovers(client, dry_run: bool = False) -> None:
    """Repair leftovers from a previously crashed run before touching anything.

    A crash mid-swap leaves {name}__migrate_backup (the intact original) and/or
    {name}__migrate_tmp (the re-embedded copy). Restore instead of deleting:
      - backup -> name when name is missing (crash between the two renames)
      - leftover backup removed only when name exists (crash after the swap completed)
      - tmp promoted to name only when BOTH name and backup are missing
        (legacy delete-then-rename crash where the tmp is the sole copy)
    """
    existing = set(_collection_names(client))
    for cname in sorted(existing):
        if not cname.endswith(BACKUP_SUFFIX):
            continue
        base = cname[: -len(BACKUP_SUFFIX)]
        if base not in existing:
            print(f"Recovery: {'would restore' if dry_run else 'restoring'} '{base}' from '{cname}' (crashed mid-swap)")
            if not dry_run:
                client.get_collection(cname).modify(name=base)
                existing.discard(cname)
                existing.add(base)
        else:
            print(f"Recovery: '{base}' exists — {'would remove' if dry_run else 'removing'} swap leftover '{cname}'")
            if not dry_run:
                client.delete_collection(cname)
                existing.discard(cname)
    for cname in sorted(existing):
        if not cname.endswith(TMP_SUFFIX):
            continue
        base = cname[: -len(TMP_SUFFIX)]
        if base not in existing and f"{base}{BACKUP_SUFFIX}" not in existing:
            print(
                f"Recovery: {'would promote' if dry_run else 'promoting'} '{cname}' to '{base}' (sole surviving copy)"
            )
            if not dry_run:
                client.get_collection(cname).modify(name=base)
                existing.discard(cname)
                existing.add(base)


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-embed all ChromaDB collections with the configured model.")
    parser.add_argument("--model", default=None, help="Embedding model ID (overrides RAG_EMBEDDING_MODEL)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing")
    args = parser.parse_args()

    from config import settings

    if args.model:
        # Must happen before the retriever singleton loads: model + embed-cache path both key off it
        settings.embedding_model = args.model

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    from core.embed_cache import get_embed_cache
    from core.retriever import embed_texts, get_embedding_model

    print(f"Loading embedding model: {settings.embedding_model}")
    model = get_embedding_model()
    target_dim = model.get_sentence_embedding_dimension()
    print(f"Target dimension: {target_dim}")

    persist_dir = settings.chroma_persist_dir
    if not Path(persist_dir).exists():
        print(f"No ChromaDB persist dir at {persist_dir} — nothing to migrate.")
        return 0

    client = chromadb.PersistentClient(path=persist_dir, settings=ChromaSettings(anonymized_telemetry=False))
    _recover_leftovers(client, dry_run=args.dry_run)
    names = [n for n in _collection_names(client) if not n.endswith((TMP_SUFFIX, BACKUP_SUFFIX))]
    if not names:
        print(f"No collections found in {persist_dir} — nothing to migrate.")
        return 0

    print(f"Found {len(names)} collection(s) in {persist_dir}\n")

    if args.dry_run:
        total = 0
        for name in names:
            collection = client.get_collection(name)
            count = collection.count()
            stored_dim = _stored_dimension(collection)
            status = "empty" if count == 0 else f"{stored_dim} -> {target_dim} dims"
            print(f"  {name}: {count} docs ({status})")
            total += count
        print(f"\nDry run: {total} document(s) would be re-embedded with '{settings.embedding_model}'.")
        return 0

    start = time.time()
    total = 0
    for name in names:
        total += _migrate_collection(client, name, embed_texts)
    get_embed_cache().flush()

    print(f"\nDone: {total} document(s) re-embedded across {len(names)} collection(s) in {time.time() - start:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
