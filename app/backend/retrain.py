"""Rules for retraining: which reviewed images may be used, when a round starts, and the bookkeeping of a round.
The training itself is a separate function that receives the selected rows; it never reads the split manifests'
test or external images, because anything whose md5 is blocked is dropped here first."""
from .db import now, reviewed_unused


def blocked_md5(manifest_path, splits=("test", "external")):
    """md5 of every image of the blocked splits, from the tracked split manifest (tsv with md5 and split columns)."""
    if not manifest_path:
        return set()
    rows = [l.rstrip("\n").split("\t") for l in open(manifest_path, encoding="utf-8")]
    head = rows[0]
    i_md5, i_split = head.index("md5"), head.index("split")
    return {r[i_md5] for r in rows[1:] if r[i_split] in splits and r[i_md5] not in ("", "-")}


def select_training_rows(conn, settings):
    """Reviewed, unused images that are not in the blocked splits. Returns (rows, n_blocked)."""
    blocked = blocked_md5(settings.blocked_manifest, settings.blocked_splits)
    rows = reviewed_unused(conn)
    keep = [r for r in rows if r["image_md5"] not in blocked]
    return keep, len(rows) - len(keep)


def ready(conn, settings):
    rows, _ = select_training_rows(conn, settings)
    return len(rows) >= settings.retrain_min


def start_round(conn, settings, model_from):
    """Claim the selected images for a new round. Returns (round_id, rows) or (None, [])."""
    rows, _ = select_training_rows(conn, settings)
    if len(rows) < settings.retrain_min:
        return None, []
    cur = conn.execute("INSERT INTO retrain_rounds (model_from, n_images, status, started_at) VALUES (?, ?, 'running', ?)",
                       (model_from, len(rows), now()))
    round_id = cur.lastrowid
    conn.executemany("UPDATE predictions SET round_id = ? WHERE id = ?", [(round_id, r["id"]) for r in rows])
    conn.commit()
    return round_id, rows


def finish_round(conn, round_id, model_to, ok=True):
    conn.execute("UPDATE retrain_rounds SET status = ?, model_to = ?, finished_at = ? WHERE id = ?",
                 ("done" if ok else "failed", model_to if ok else None, now(), round_id))
    if not ok:  # a failed round releases its images for the next one
        conn.execute("UPDATE predictions SET round_id = NULL WHERE round_id = ?", (round_id,))
    conn.commit()
