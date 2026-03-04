"""
Document ingestion pipeline with code-aware chunking + multimodal support.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

from config import settings
from core.multimodal import is_multimodal_file, extract_multimodal

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    # Python
    ".py", ".pyi", ".pyw",
    # JavaScript / TypeScript
    ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    # Web
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".vue", ".svelte",
    # Java / Kotlin / Scala
    ".java", ".kt", ".kts", ".scala", ".groovy", ".gradle",
    # C / C++ / C#
    ".c", ".cpp", ".h", ".hpp", ".cs", ".csx",
    # Systems
    ".go", ".rs", ".swift", ".m", ".mm",
    # Scripting
    ".rb", ".php", ".pl", ".pm", ".lua", ".r", ".R",
    # Shell
    ".sh", ".bash", ".zsh", ".fish", ".bat", ".cmd", ".ps1",
    # Config / Data
    ".yaml", ".yml", ".json", ".toml", ".xml", ".ini", ".cfg", ".conf",
    ".env", ".properties", ".gradle",
    # Docs
    ".md", ".mdx", ".rst", ".txt", ".adoc", ".tex", ".log",
    # SQL / DB
    ".sql", ".graphql", ".gql",
    # DevOps
    ".dockerfile", ".tf", ".hcl",
    # Multimodal
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp",
    # Misc code
    ".dart", ".ex", ".exs", ".erl", ".hs", ".clj", ".lisp", ".elm",
    ".proto", ".thrift", ".avsc",
}

LANG_MAP = {
    ".py": "python", ".pyi": "python", ".pyw": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".jsx": "javascript", ".tsx": "typescript",
    ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss",
    ".sass": "sass", ".less": "less", ".vue": "vue", ".svelte": "svelte",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala",
    ".groovy": "groovy", ".gradle": "groovy",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
    ".cs": "csharp", ".csx": "csharp",
    ".go": "go", ".rs": "rust", ".swift": "swift", ".m": "objc", ".mm": "objc",
    ".rb": "ruby", ".php": "php", ".pl": "perl", ".pm": "perl",
    ".lua": "lua", ".r": "r", ".R": "r",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".fish": "bash",
    ".bat": "batch", ".cmd": "batch", ".ps1": "powershell",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml",
    ".xml": "xml", ".ini": "ini", ".cfg": "ini", ".conf": "ini",
    ".env": "text", ".properties": "properties",
    ".md": "markdown", ".mdx": "markdown", ".rst": "rst",
    ".txt": "text", ".adoc": "asciidoc", ".tex": "latex", ".log": "text",
    ".sql": "sql", ".graphql": "graphql", ".gql": "graphql",
    ".dockerfile": "docker", ".tf": "hcl", ".hcl": "hcl",
    ".pdf": "pdf", ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".gif": "image", ".webp": "image",
    ".dart": "dart", ".ex": "elixir", ".exs": "elixir", ".erl": "erlang",
    ".hs": "haskell", ".clj": "clojure", ".lisp": "lisp", ".elm": "elm",
    ".proto": "protobuf", ".thrift": "thrift", ".avsc": "json",
}

SKIP_DIRS = {
    "node_modules", "__pycache__", "venv", ".venv", ".git", ".next", "dist",
    "build", ".tox", "env", ".idea", ".vscode", ".settings", ".gradle",
    "target", "bin", "obj", ".svn", ".hg", "vendor", "coverage",
    ".pytest_cache", ".mypy_cache", "eggs", "*.egg-info",
}


@dataclass
class Document:
    content: str
    filepath: str
    language: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    content: str
    chunk_id: str
    document_path: str
    language: str
    start_line: int
    end_line: int
    chunk_type: str
    metadata: dict = field(default_factory=dict)

    @property
    def display_source(self):
        return "{}:{}-{}".format(self.document_path, self.start_line, self.end_line)


_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(_tokenizer.encode(text, disallowed_special=()))


def _should_skip(filepath, root):
    """Check if a file should be skipped based on directory names.
    Fixed for Windows paths where parts[0] can be 'D:\\'."""
    try:
        rel = filepath.relative_to(root)
        parts = rel.parts
    except ValueError:
        parts = filepath.parts

    for p in parts[:-1]:  # Check directory parts only (not the filename)
        if p in SKIP_DIRS or p.startswith("."):
            return True
    return False


def load_documents(directory):
    """Recursively load all supported documents."""
    docs = []
    root = Path(directory)
    if not root.exists():
        logger.warning("Directory not found: {}".format(directory))
        return docs

    for filepath in sorted(root.rglob("*")):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if _should_skip(filepath, root):
            continue

        rel_path = str(filepath.relative_to(root)).replace("\\", "/")
        lang = LANG_MAP.get(filepath.suffix.lower(), "text")

        try:
            if is_multimodal_file(str(filepath)):
                extracted = extract_multimodal(str(filepath))
                if extracted and extracted.text.strip():
                    docs.append(Document(
                        content=extracted.text,
                        filepath=rel_path,
                        language=lang,
                        metadata={
                            "multimodal": True,
                            "image_count": len(extracted.images),
                        },
                    ))
                    logger.info("Loaded (multimodal): {} ({})".format(rel_path, lang))
                continue

            content = filepath.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                continue
            docs.append(Document(content=content, filepath=rel_path, language=lang))
            logger.info("Loaded: {} ({}, {} tokens)".format(rel_path, lang, count_tokens(content)))

        except Exception as e:
            logger.error("Failed to load {}: {}".format(filepath, e))

    logger.info("Loaded {} documents from {}".format(len(docs), directory))
    return docs


def load_single_file(filepath, base_dir=""):
    """Load a single file."""
    p = Path(filepath)
    if not p.is_file():
        return None
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None

    lang = LANG_MAP.get(p.suffix.lower(), "text")
    rel = str(p.relative_to(base_dir)).replace("\\", "/") if base_dir else p.name

    try:
        if is_multimodal_file(filepath):
            extracted = extract_multimodal(filepath)
            if extracted and extracted.text.strip():
                return Document(
                    content=extracted.text, filepath=rel, language=lang,
                    metadata={"multimodal": True},
                )
            return None

        content = p.read_text(encoding="utf-8", errors="replace")
        return Document(content=content, filepath=rel, language=lang)
    except Exception as e:
        logger.error("Failed to load {}: {}".format(filepath, e))
        return None


# ── Chunking ──

def _generate_chunk_id(path, start, end):
    raw = "{}:{}:{}".format(path, start, end)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _chunk_code(doc):
    lines = doc.content.split("\n")
    chunks = []

    block_patterns = {
        "python": re.compile(r"^(class |def |async def |@)"),
        "javascript": re.compile(r"^(function |class |const |let |var |export |import )"),
        "typescript": re.compile(r"^(function |class |const |let |var |export |import |interface |type )"),
        "go": re.compile(r"^(func |type |var |const |package )"),
        "rust": re.compile(r"^(fn |pub fn |struct |enum |impl |mod |use )"),
        "java": re.compile(r"^(public |private |protected |class |interface |enum )"),
        "ruby": re.compile(r"^(def |class |module |require )"),
        "c": re.compile(r"^(int |void |char |float |double |struct |enum |typedef |#include )"),
        "cpp": re.compile(r"^(int |void |char |class |struct |enum |template |namespace |#include )"),
        "csharp": re.compile(r"^(public |private |protected |class |interface |enum |namespace |using )"),
    }

    pattern = block_patterns.get(doc.language)
    if not pattern:
        return _chunk_prose(doc)

    boundaries = [0]
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and pattern.match(stripped):
            if i > 0 and i != boundaries[-1]:
                boundaries.append(i)
    boundaries.append(len(lines))

    buffer_start = boundaries[0]
    buffer_lines = []

    for i in range(len(boundaries) - 1):
        block = lines[boundaries[i]: boundaries[i + 1]]
        block_tokens = count_tokens("\n".join(block))

        if buffer_lines and count_tokens("\n".join(buffer_lines)) + block_tokens > settings.chunk_size:
            content = "\n".join(buffer_lines)
            if count_tokens(content) >= settings.min_chunk_size:
                chunks.append(Chunk(
                    content=content,
                    chunk_id=_generate_chunk_id(doc.filepath, buffer_start, boundaries[i] - 1),
                    document_path=doc.filepath, language=doc.language,
                    start_line=buffer_start + 1, end_line=boundaries[i],
                    chunk_type="code",
                ))
            buffer_start = boundaries[i]
            buffer_lines = block
        else:
            buffer_lines.extend(block)

    if buffer_lines:
        content = "\n".join(buffer_lines)
        if count_tokens(content) >= settings.min_chunk_size:
            chunks.append(Chunk(
                content=content,
                chunk_id=_generate_chunk_id(doc.filepath, buffer_start, len(lines)),
                document_path=doc.filepath, language=doc.language,
                start_line=buffer_start + 1, end_line=len(lines),
                chunk_type="code",
            ))

    return chunks


def _chunk_prose(doc):
    lines = doc.content.split("\n")
    chunks = []

    sections = []
    current_start = 0
    current_lines = []

    for i, line in enumerate(lines):
        is_heading = line.strip().startswith("#") or (
            i > 0 and lines[i - 1].strip() and re.match(r"^[=\-]{3,}$", line.strip())
        )
        if is_heading and current_lines:
            sections.append((current_start, current_lines))
            current_start = i
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_start, current_lines))

    for start_line, section_lines in sections:
        section_text = "\n".join(section_lines)
        token_count = count_tokens(section_text)

        if token_count <= settings.chunk_size:
            if token_count >= settings.min_chunk_size:
                chunks.append(Chunk(
                    content=section_text,
                    chunk_id=_generate_chunk_id(doc.filepath, start_line, start_line + len(section_lines)),
                    document_path=doc.filepath, language=doc.language,
                    start_line=start_line + 1, end_line=start_line + len(section_lines),
                    chunk_type="prose",
                ))
        else:
            window = []
            window_start = start_line
            for j, sl in enumerate(section_lines):
                window.append(sl)
                if count_tokens("\n".join(window)) >= settings.chunk_size:
                    content = "\n".join(window)
                    chunks.append(Chunk(
                        content=content,
                        chunk_id=_generate_chunk_id(doc.filepath, window_start, start_line + j),
                        document_path=doc.filepath, language=doc.language,
                        start_line=window_start + 1, end_line=start_line + j + 1,
                        chunk_type="prose",
                    ))
                    overlap_tokens = 0
                    overlap_start = len(window)
                    for k in range(len(window) - 1, -1, -1):
                        overlap_tokens += count_tokens(window[k])
                        if overlap_tokens >= settings.chunk_overlap:
                            overlap_start = k
                            break
                    window = window[overlap_start:]
                    window_start = start_line + j - len(window) + 1

            if window and count_tokens("\n".join(window)) >= settings.min_chunk_size:
                content = "\n".join(window)
                chunks.append(Chunk(
                    content=content,
                    chunk_id=_generate_chunk_id(doc.filepath, window_start, start_line + len(section_lines)),
                    document_path=doc.filepath, language=doc.language,
                    start_line=window_start + 1, end_line=start_line + len(section_lines),
                    chunk_type="prose",
                ))

    return chunks


CODE_LANGUAGES = {"python", "javascript", "typescript", "go", "rust", "java", "ruby", "bash", "c", "cpp", "csharp"}


def chunk_document(doc):
    if doc.language in CODE_LANGUAGES:
        chunks = _chunk_code(doc)
    else:
        chunks = _chunk_prose(doc)

    for chunk in chunks:
        chunk.metadata["language"] = doc.language
        chunk.metadata["source_file"] = doc.filepath
        chunk.metadata.update(doc.metadata)

    logger.info("Chunked {} -> {} chunks".format(doc.filepath, len(chunks)))
    return chunks


def ingest_directory(directory):
    docs = load_documents(directory)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    logger.info("Total chunks: {}".format(len(all_chunks)))
    return all_chunks