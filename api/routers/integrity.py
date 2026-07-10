"""Knowledge integrity scan endpoints."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from api import database as db
from api.auth import require_auth
from api.dependencies import get_user_store
from api.models import IntegrityScanReq
from core.integrity import run_integrity_scan

router = APIRouter(prefix="/api/integrity", tags=["integrity"])


@router.post("/scan")
async def integrity_scan(req: IntegrityScanReq, user=Depends(require_auth)):
    uid = user["id"]
    s = await asyncio.to_thread(get_user_store, uid)
    if s.count == 0:
        raise HTTPException(400, "No documents indexed")
    prev = await asyncio.to_thread(db.get_latest_integrity_fingerprints)
    result = await asyncio.to_thread(run_integrity_scan, s, previous_fingerprints=prev)
    if req.persist:
        scan_id = await asyncio.to_thread(db.save_integrity_scan, uid, result)
        result["scan_id"] = scan_id
    return result


@router.get("/history")
def integrity_history(days: int = 30, limit: int = 30):
    return db.get_integrity_history(days=days, limit=limit)


@router.get("/scan/{scan_id}")
def integrity_scan_detail(scan_id: str):
    res = db.get_integrity_scan(scan_id)
    if not res:
        raise HTTPException(404, "Scan not found")
    return res
