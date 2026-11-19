"""Accounts: usernames and bcrypt password hashes. No plaintext password is ever stored or logged."""
import re
import sqlite3
import bcrypt

from .db import now

USERNAME = re.compile(r"^[A-Za-z0-9._-]{3,32}$")
MIN_PASSWORD = 8
_DUMMY = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt())  # compared when the user is unknown, so timing does not reveal it


def register(conn, username, password):
    """Returns (ok, message)."""
    if not USERNAME.match(username or ""):
        return False, "Username: 3 to 32 characters, letters, digits, dot, dash or underscore."
    if len(password or "") < MIN_PASSWORD:
        return False, "Password: at least %d characters." % MIN_PASSWORD
    raw = password.encode("utf-8")
    if len(raw) > 72:
        return False, "Password: at most 72 bytes."  # bcrypt ignores everything after 72 bytes
    try:
        conn.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                     (username, bcrypt.hashpw(raw, bcrypt.gensalt()), now()))
        conn.commit()
    except sqlite3.IntegrityError:
        return False, "That username is taken."
    return True, "Account created."


def login(conn, username, password):
    """The user id, or None. Same answer for an unknown user and a wrong password."""
    row = conn.execute("SELECT id, password_hash FROM users WHERE username = ?", (username or "",)).fetchone()
    raw = (password or "").encode("utf-8")[:72]
    stored = row["password_hash"] if row else _DUMMY
    ok = bcrypt.checkpw(raw, stored)
    return row["id"] if (row and ok) else None
