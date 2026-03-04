import argparse
import logging
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from config import settings
from core.ingestion import ingest_directory
from core.retriever import VectorStore, retrieve
from core.generator import generate, Message
from api.database import init_db

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("rag.log"), logging.StreamHandler()])


def cmd_ingest(args):
    store = VectorStore()
    chunks = ingest_directory(args.directory)
    if not chunks:
        return console.print("[red]No documents found![/red]")
    added = store.add_chunks(chunks)
    t = Table(title="Ingestion Summary")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="green")
    t.add_row("Documents", str(len(set(c.document_path for c in chunks))))
    t.add_row("Chunks", str(added))
    t.add_row("Total", str(store.count))
    console.print(t)


def cmd_query(args):
    store = VectorStore()
    if store.count == 0:
        return console.print("[red]No docs indexed.[/red]")
    hits = retrieve(store=store, query=args.query, use_hybrid=True)
    resp = generate(query=args.query, hits=hits)
    console.print(Panel(Markdown(resp.answer), title="Answer", border_style="green"))


def cmd_chat(args):
    store = VectorStore()
    if store.count == 0:
        return console.print("[red]No docs indexed.[/red]")
    console.print(Panel("[bold]RAG Chat[/bold] - type 'quit' to exit", border_style="blue"))
    history = []
    while True:
        try:
            q = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        if q == "/clear":
            history.clear()
            console.print("[dim]Cleared.[/dim]")
            continue
        with console.status("[green]Thinking..."):
            hits = retrieve(store=store, query=q, use_hybrid=True)
            r = generate(query=q, hits=hits, conversation_history=history)
        console.print(Panel(Markdown(r.answer), title="Assistant", border_style="green"))
        history.extend([Message("user", q), Message("assistant", r.answer)])


def cmd_serve(args):
    import uvicorn
    init_db()
    port = args.port or settings.api_port
    console.print(f"[bold blue]Starting API: http://localhost:{port}[/bold blue]")
    uvicorn.run("api.server:app", host=settings.api_host, port=port, reload=args.reload)


def main():
    p = argparse.ArgumentParser(description="RAG Assistant v2")
    sp = p.add_subparsers(dest="command")
    i = sp.add_parser("ingest")
    i.add_argument("directory")
    q = sp.add_parser("query")
    q.add_argument("query")
    sp.add_parser("chat")
    s = sp.add_parser("serve")
    s.add_argument("--port", type=int)
    s.add_argument("--reload", action="store_true")
    args = p.parse_args()
    cmds = {"ingest": cmd_ingest, "query": cmd_query, "chat": cmd_chat, "serve": cmd_serve}
    if args.command in cmds:
        cmds[args.command](args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
