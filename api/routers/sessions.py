"""Session CRUD endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from api import database as db
from api.auth import require_auth
from api.models import RenameSessionReq

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _owned_session(sid: str, user: dict) -> dict:
    """Return the session if it belongs to the user, else 404 (avoids leaking existence)."""
    sess = db.get_session(sid)
    if not sess or sess["user_id"] != user["id"]:
        raise HTTPException(404, "Session not found")
    return sess


@router.post("")
def create_session(user=Depends(require_auth)):
    return db.create_session(user["id"])


@router.get("")
def list_sessions(user=Depends(require_auth)):
    return {"sessions": db.get_user_sessions(user["id"])}


@router.get("/{sid}/messages")
def get_messages(sid: str, user=Depends(require_auth)):
    _owned_session(sid, user)
    return {"messages": db.get_session_messages(sid)}


@router.put("/{sid}")
def rename_session(sid: str, body: RenameSessionReq, user=Depends(require_auth)):
    _owned_session(sid, user)
    db.update_session_title(sid, body.title)
    return {"status": "ok"}


@router.delete("/{sid}")
def del_session(sid: str, user=Depends(require_auth)):
    _owned_session(sid, user)
    db.delete_session(sid)
    return {"status": "deleted"}
