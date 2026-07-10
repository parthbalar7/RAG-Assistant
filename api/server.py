"""
RAG Assistant v2 — FastAPI application entry point.
All endpoint logic is in api/routers/. This file handles app setup only.
"""

import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import database as db
from api.helpers import get_maintenance_candidates, mark_maintenance_done
from api.routers import (
    auth,
    graph,
    ingest,
    integrity,
    llm,
    memory,
    pageindex,
    query,
    sessions,
    websocket,
)
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def _memory_maintenance_loop():
    """Idle-time memory upkeep: every sweep period, run consolidation/archiving/style-rule
    distillation for each user idle past the threshold with memory writes since their last
    sweep. Per-user failures (e.g. Ollama unreachable) are logged and never kill the loop."""
    from core.memory import run_maintenance

    while True:
        await asyncio.sleep(settings.memory_maintenance_interval_s)
        try:
            candidates = get_maintenance_candidates(settings.memory_idle_threshold_s)
        except Exception as e:
            logger.warning("Maintenance candidate scan failed: %s", e)
            continue
        for uid in candidates:
            started = time.time()
            try:
                counts = await asyncio.to_thread(run_maintenance, uid)
                mark_maintenance_done(uid, started)
                logger.info("Idle memory maintenance for '%s': %s", uid, counts)
            except Exception as e:
                logger.warning("Idle memory maintenance failed for '%s': %s", uid, e)


@asynccontextmanager
async def lifespan(app):
    db.init_db()
    maintenance_task = asyncio.create_task(_memory_maintenance_loop()) if settings.memory_enabled else None
    logger.info("RAG server ready")
    try:
        yield
    finally:
        if maintenance_task is not None:
            maintenance_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await maintenance_task


app = FastAPI(title="RAG Assistant v2", version="2.0.0", lifespan=lifespan)

# CORS — use configured origins (defaults to localhost:3000 for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──
app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(websocket.router)
app.include_router(integrity.router)
app.include_router(graph.router)
app.include_router(llm.router)
app.include_router(memory.router)
app.include_router(pageindex.router)

# ── Serve frontend build if available ──
_fb = Path(__file__).parent.parent / "frontend" / "build"
if not _fb.exists():
    _fb = Path(__file__).parent.parent / "build"
if _fb.exists():
    app.mount("/", StaticFiles(directory=str(_fb), html=True), name="frontend")
