"""SQLite storage: users, predictions and retraining rounds. Plain sqlite3, parameterised queries only."""
import json
import sqlite3
from datetime import datetime, timezone

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
STATUSES = ("saved", "pending", "confirmed", "corrected")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS retrain_rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_from TEXT NOT NULL,
    model_to TEXT,
    n_images INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    image_path TEXT NOT NULL,
    image_md5 TEXT NOT NULL,
    probs TEXT NOT NULL,
    predicted TEXT NOT NULL,
    margin REAL NOT NULL,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('saved', 'pending', 'confirmed', 'corrected')),
    true_class TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    round_id INTEGER REFERENCES retrain_rounds(id)
);
CREATE INDEX IF NOT EXISTS idx_pred_user ON predictions(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_pred_queue ON predictions(status, margin);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def margin_of(probs):
    """Difference between the two highest class probabilities (low = the model is unsure)."""
    top = sorted(probs.values(), reverse=True)
    return float(top[0] - top[1])


def add_prediction(conn, user_id, image_path, image_md5, probs, model_version, send_to_review):
    assert set(probs) == set(CLASSES), "probabilities must cover the four classes"
    predicted = max(probs, key=probs.get)
    cur = conn.execute(
        "INSERT INTO predictions (user_id, image_path, image_md5, probs, predicted, margin, model_version, status, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, image_path, image_md5, json.dumps(probs), predicted, margin_of(probs), model_version,
         "pending" if send_to_review else "saved", now()))
    conn.commit()
    return cur.lastrowid


def find_duplicate(conn, user_id, image_md5):
    """An earlier prediction of the same file by the same user, or None (exact duplicates only)."""
    return conn.execute("SELECT * FROM predictions WHERE user_id = ? AND image_md5 = ? ORDER BY id LIMIT 1",
                        (user_id, image_md5)).fetchone()


def send_to_review(conn, pred_id, user_id):
    cur = conn.execute("UPDATE predictions SET status = 'pending' WHERE id = ? AND user_id = ? AND status = 'saved'",
                       (pred_id, user_id))
    conn.commit()
    return cur.rowcount == 1


def queue(conn, user_id):
    """Pending images, the most uncertain (lowest margin) first."""
    return conn.execute("SELECT * FROM predictions WHERE user_id = ? AND status = 'pending' ORDER BY margin ASC, id ASC",
                        (user_id,)).fetchall()


def review(conn, pred_id, user_id, true_class, notes=""):
    """Confirm (true class equals the prediction) or correct a pending prediction."""
    if true_class not in CLASSES:
        raise ValueError("unknown class: %r" % (true_class,))
    row = conn.execute("SELECT predicted FROM predictions WHERE id = ? AND user_id = ? AND status = 'pending'",
                       (pred_id, user_id)).fetchone()
    if row is None:
        return False
    status = "confirmed" if row["predicted"] == true_class else "corrected"
    conn.execute("UPDATE predictions SET status = ?, true_class = ?, notes = ?, reviewed_at = ? WHERE id = ?",
                 (status, true_class, notes, now(), pred_id))
    conn.commit()
    return True


def history(conn, user_id, query="", cls=None, status=None):
    sql, args = "SELECT * FROM predictions WHERE user_id = ?", [user_id]
    if query:
        sql += " AND image_path LIKE ? ESCAPE '\\'"
        args.append("%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
    if cls:
        sql += " AND predicted = ?"
        args.append(cls)
    if status:
        sql += " AND status = ?"
        args.append(status)
    return conn.execute(sql + " ORDER BY created_at DESC, id DESC", args).fetchall()


def delete_prediction(conn, pred_id, user_id):
    cur = conn.execute("DELETE FROM predictions WHERE id = ? AND user_id = ? AND round_id IS NULL", (pred_id, user_id))
    conn.commit()
    return cur.rowcount == 1


def count_pending(conn, user_id, margin_threshold=None):
    sql, args = "SELECT COUNT(*) FROM predictions WHERE user_id = ? AND status = 'pending'", [user_id]
    if margin_threshold is not None:
        sql += " AND margin < ?"
        args.append(margin_threshold)
    return conn.execute(sql, args).fetchone()[0]


def reviewed_unused(conn):
    """Reviewed images (any user) not yet used in a retraining round."""
    return conn.execute("SELECT * FROM predictions WHERE status IN ('confirmed', 'corrected') AND round_id IS NULL"
                        " ORDER BY id").fetchall()
