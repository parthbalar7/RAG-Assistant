"""JWT authentication and password hashing utilities."""

import logging
import time

import bcrypt
import jwt
from fastapi import Header, HTTPException

from api import database as db
from config import settings

logger = logging.getLogger(__name__)


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    return bcrypt.checkpw(pw.encode(), hashed.encode())


def create_token(uid: str, username: str) -> str:
    return jwt.encode(
        {"sub": uid, "username": username, "exp": time.time() + settings.jwt_expiry_hours * 3600},
        settings.jwt_secret,
        algorithm="HS256",
    )


def decode_token(token: str) -> dict | None:
    try:
        p = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        return p if p.get("exp", 0) > time.time() else None
    except jwt.ExpiredSignatureError:
        logger.debug("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.debug("Invalid token: %s", e)
        return None
    except Exception as e:
        logger.warning("Unexpected token decode error: %s", e)
        return None


def get_current_user(authorization: str = Header(default=None)) -> dict | None:
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
    except Exception as e:
        logger.warning("get_current_user error: %s", e)
        return None


def require_auth(authorization: str = Header(default=None)) -> dict:
    u = get_current_user(authorization)
    if not u:
        raise HTTPException(401, "Not authenticated")
    return u
