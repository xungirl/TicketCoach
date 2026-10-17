"""
Per-trainee identity via HMAC-signed tokens (stdlib only, no user table).

Flow: POST /api/login {name, access_code}  ->  {"token": "<id>.<expiry>.<sig>"}
The trainee id is derived from the name; the signature proves the server
issued it, so nobody can read another trainee's memory by sending their id.

Why not JWT: the token has exactly two claims (id, expiry) and one issuer —
a 20-line HMAC is easier to audit than a dependency, and it's the same idea.
"""

import base64
import hashlib
import hmac
import os
import re
import secrets
import time
from typing import Optional

TOKEN_TTL_S = 30 * 24 * 3600

_dev_secret = secrets.token_hex(32)  # fallback: tokens die on restart (fine for dev)


def _secret() -> bytes:
    return (os.environ.get("AUTH_SECRET") or _dev_secret).encode()


def trainee_id_for(name: str) -> str:
    """Stable, URL-safe id for a display name ("张三" -> "t_ab12cd34ef")."""
    norm = re.sub(r"\s+", " ", name).strip().lower()
    if not norm:
        raise ValueError("name required")
    return "t_" + hashlib.sha256(norm.encode()).hexdigest()[:10]


def _sign(payload: str) -> str:
    mac = hmac.new(_secret(), payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac[:18]).decode().rstrip("=")


def issue_token(trainee_id: str, ttl_s: int = TOKEN_TTL_S) -> str:
    payload = f"{trainee_id}.{int(time.time()) + ttl_s}"
    return f"{payload}.{_sign(payload)}"


def verify_token(token: Optional[str]) -> Optional[str]:
    """trainee_id if the token is authentic and unexpired, else None."""
    if not token or token.count(".") != 2:
        return None
    tid, exp, sig = token.split(".")
    if not hmac.compare_digest(sig, _sign(f"{tid}.{exp}")):
        return None
    try:
        if int(exp) < time.time():
            return None
    except ValueError:
        return None
    return tid
