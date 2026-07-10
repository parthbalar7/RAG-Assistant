"""Auth endpoints: register, login, me."""

from fastapi import APIRouter, Depends, HTTPException

from api import database as db
from api.auth import create_token, get_current_user, hash_password, verify_password
from api.models import AuthReq

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register")
def register(req: AuthReq):
    if db.get_user_by_username(req.username):
        raise HTTPException(400, "Username exists")
    u = db.create_user(req.username, hash_password(req.password), req.display_name)
    return {"token": create_token(u["id"], req.username), "user": u}


@router.post("/login")
def login(req: AuthReq):
    u = db.get_user_by_username(req.username)
    if not u or not verify_password(req.password, u["password_hash"]):
        raise HTTPException(401, "Invalid credentials")
    return {
        "token": create_token(u["id"], u["username"]),
        "user": {"id": u["id"], "username": u["username"], "display_name": u["display_name"]},
    }


@router.get("/me")
def me(user=Depends(get_current_user)):
    if not user:
        return {"user": None}
    return {"user": {"id": user["id"], "username": user["username"], "display_name": user["display_name"]}}
