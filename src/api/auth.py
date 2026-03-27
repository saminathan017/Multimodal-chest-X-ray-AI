"""
src/api/auth.py — JWT Authentication & Role-Based Access
"""
from __future__ import annotations
import os
import time
from enum import Enum
from typing import Optional

from loguru import logger

try:
    import jwt as pyjwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT not installed. Auth running in mock mode.")


class UserRole(str, Enum):
    CLINICIAN  = "clinician"
    RADIOLOGIST= "radiologist"
    ADMIN      = "admin"
    RESEARCHER = "researcher"


JWT_SECRET   = os.getenv("JWT_SECRET", "CHANGE-ME-IN-PRODUCTION-use-32-char-random-key")
JWT_ALGO     = "HS256"
JWT_EXPIRY_H = 8   # 8-hour tokens for clinical sessions


class JWTHandler:
    def create_token(self, user_id: str, role: UserRole, institution: str) -> str:
        payload = {
            "sub":         user_id,
            "role":        role.value,
            "institution": institution,
            "iat":         int(time.time()),
            "exp":         int(time.time()) + JWT_EXPIRY_H * 3600,
        }
        if JWT_AVAILABLE:
            return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
        return "mock-token"

    def verify_token(self, token: str) -> Optional[dict]:
        if not JWT_AVAILABLE:
            return {"sub": "mock-user", "role": "admin", "institution": "demo"}
        try:
            return pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        except Exception:
            return None


def decode_token(token: str) -> Optional[dict]:
    handler = JWTHandler()
    return handler.verify_token(token)
