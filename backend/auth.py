"""
BlockVerify — JWT Authentication Module
========================================
Provides:
  POST /api/auth/register  → create account, return token
  POST /api/auth/login     → verify password, return token
  GET  /api/auth/me        → validate token, return username

  require_auth(fn)         → decorator that enforces a valid Bearer token
                             and sets flask.g.user to the authenticated username

Passwords are hashed with bcrypt (cost=12).
Tokens are signed with HS256 / 24-hour expiry.
The signing secret is auto-generated once and persisted to data/secret.key.
"""

import os
import json
import secrets
from functools import wraps
from time import time

import bcrypt
import jwt
from flask import Blueprint, request, jsonify, g

# ── Blueprint ────────────────────────────────────────────────────────────────
auth_bp = Blueprint("auth", __name__)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
SECRET_FILE = os.path.join(DATA_DIR, "secret.key")

os.makedirs(DATA_DIR, exist_ok=True)

# ── Secret key (generated once, persisted) ───────────────────────────────────
def _lock_down(path: str) -> None:
    """Restrict the secret file to owner read/write only (chmod 600). No-op on Windows."""
    try:
        os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass  # non-POSIX filesystem (e.g. Windows) — best effort only

def _load_secret() -> str:
    if os.path.exists(SECRET_FILE):
        _lock_down(SECRET_FILE)          # ensure existing key is owner-only
        with open(SECRET_FILE) as f:
            return f.read().strip()
    key = secrets.token_hex(48)          # 384-bit random key
    # Create the file with 0600 perms BEFORE writing the secret (avoids a brief world-readable window)
    fd = os.open(SECRET_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key)
    _lock_down(SECRET_FILE)
    return key

SECRET_KEY = _load_secret()
TOKEN_TTL  = 86_400          # 24 hours in seconds
BCRYPT_COST = 12

# ── User store helpers ────────────────────────────────────────────────────────
def _load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

# ── Token helpers ─────────────────────────────────────────────────────────────
def _make_token(username: str) -> str:
    payload = {
        "sub": username,
        "iat": int(time()),
        "exp": int(time()) + TOKEN_TTL,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def _decode_token(token: str) -> dict:
    """Return decoded payload or raise jwt.PyJWTError."""
    return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])


def decode_token(token: str) -> dict | None:
    """Public helper: decode a token and return payload, or None on failure."""
    try:
        return _decode_token(token)
    except jwt.PyJWTError:
        return None

# ── require_auth decorator ────────────────────────────────────────────────────
def require_auth(fn):
    """
    Decorator for Flask route functions that require a valid JWT.

    On success  → sets g.user to the authenticated username and calls fn(*a, **kw).
    On failure  → returns JSON 401.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"success": False, "error": "Authentication required. Please log in."}), 401
        token = auth_header[len("Bearer "):]
        try:
            payload = _decode_token(token)
            g.user = payload["sub"]
        except jwt.ExpiredSignatureError:
            return jsonify({"success": False, "error": "Session expired. Please log in again."}), 401
        except jwt.PyJWTError:
            return jsonify({"success": False, "error": "Invalid token. Please log in again."}), 401
        return fn(*args, **kwargs)
    return wrapper

# ── Routes ───────────────────────────────────────────────────────────────────

@auth_bp.route("/api/auth/register", methods=["POST"])
def register():
    """Create a new user account and return a JWT."""
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""

    if not username:
        return jsonify({"success": False, "error": "Username is required."}), 400
    if len(username) < 3:
        return jsonify({"success": False, "error": "Username must be at least 3 characters."}), 400
    if len(username) > 32:
        return jsonify({"success": False, "error": "Username must be 32 characters or fewer."}), 400
    if not username.replace("_", "").replace("-", "").isalnum():
        return jsonify({"success": False, "error": "Username may only contain letters, numbers, hyphens, and underscores."}), 400
    if len(password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters."}), 400

    users = _load_users()
    if username in users:
        return jsonify({"success": False, "error": "Username already taken. Please choose another."}), 409

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=BCRYPT_COST)).decode()
    users[username] = {"passwordHash": pw_hash, "createdAt": int(time())}
    _save_users(users)

    token = _make_token(username)
    return jsonify({"success": True, "token": token, "username": username,
                    "message": f"Account created. Welcome, {username}!"}), 201


@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    """Verify credentials and return a JWT."""
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password are required."}), 400

    users = _load_users()
    user = users.get(username)
    if not user:
        # Deliberate vague message to prevent username enumeration
        return jsonify({"success": False, "error": "Invalid username or password."}), 401

    if not bcrypt.checkpw(password.encode(), user["passwordHash"].encode()):
        return jsonify({"success": False, "error": "Invalid username or password."}), 401

    token = _make_token(username)
    return jsonify({"success": True, "token": token, "username": username,
                    "message": f"Welcome back, {username}!"}), 200


@auth_bp.route("/api/auth/me", methods=["GET"])
@require_auth
def me():
    """Return the currently authenticated user's username."""
    return jsonify({"success": True, "username": g.user}), 200
