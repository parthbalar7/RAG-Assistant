import time
import bcrypt
import jwt
from fastapi import HTTPException, Header
from config import settings
from api import database as db


def hash_password(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_password(pw, hashed):
    return bcrypt.checkpw(pw.encode(), hashed.encode())


def create_token(uid, username):
    return jwt.encode({"sub": uid, "username": username,
                        "exp": time.time() + settings.jwt_expiry_hours * 3600},
                       settings.jwt_secret, algorithm="HS256")


def decode_token(token):
    try:
        p = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        return p if p.get("exp", 0) > time.time() else None
    except Exception:
        return None


def get_current_user(authorization: str = Header(default=None)):
    if not authorization:
        return None
    try:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return None
        p = decode_token(token)
        if not p:
            return None
        return db.get_user_by_id(p["sub"])
    except Exception:
        return None


def require_auth(authorization: str = Header(default=None)):
    u = get_current_user(authorization)
    if not u:
        raise HTTPException(401, "Not authenticated")
    return u
