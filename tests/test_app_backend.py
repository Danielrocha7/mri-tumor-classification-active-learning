"""Backend rules of the application: accounts, review queue, history, and the retraining guard. No torch needed."""
import os
import sys
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from app.backend import auth, db, retrain
from app.backend.settings import Settings


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.sqlite"))
    yield c
    c.close()


def probs(g=0.0, m=0.0, n=0.0, p=0.0):
    return {"glioma": g, "meningioma": m, "notumor": n, "pituitary": p}


def user(conn, name="alice"):
    assert auth.register(conn, name, "correct horse")[0]
    return auth.login(conn, name, "correct horse")


def add(conn, uid, name, pr, md5, review=True):
    return db.add_prediction(conn, uid, name, md5, pr, "v1", review)


def test_register_validates_username_and_password(conn):
    assert not auth.register(conn, "ab", "correct horse")[0]
    assert not auth.register(conn, "bad name", "correct horse")[0]
    assert not auth.register(conn, "alice", "short")[0]
    assert not auth.register(conn, "alice", "x" * 73)[0]
    assert auth.register(conn, "alice", "correct horse")[0]
    assert not auth.register(conn, "alice", "another password")[0]  # username taken


def test_password_is_stored_as_a_bcrypt_hash(conn):
    user(conn)
    h = conn.execute("SELECT password_hash FROM users").fetchone()[0]
    assert b"correct horse" not in h and h.startswith(b"$2")


def test_login_same_answer_for_wrong_password_and_unknown_user(conn):
    uid = user(conn)
    assert auth.login(conn, "alice", "correct horse") == uid
    assert auth.login(conn, "alice", "wrong password") is None
    assert auth.login(conn, "nobody", "correct horse") is None


def test_margin_is_gap_between_top_two(conn):
    assert db.margin_of(probs(0.52, 0.38, 0.03, 0.07)) == pytest.approx(0.14)


def test_queue_is_ordered_by_lowest_margin_first(conn):
    uid = user(conn)
    a = add(conn, uid, "a.png", probs(0.9, 0.05, 0.03, 0.02), "m1")
    b = add(conn, uid, "b.png", probs(0.45, 0.40, 0.10, 0.05), "m2")
    c = add(conn, uid, "c.png", probs(0.6, 0.3, 0.05, 0.05), "m3")
    add(conn, uid, "d.png", probs(0.5, 0.4, 0.05, 0.05), "m4", review=False)  # saved, not in the queue
    assert [r["id"] for r in db.queue(conn, uid)] == [b, c, a]


def test_review_confirms_or_corrects_and_only_once(conn):
    uid = user(conn)
    p = add(conn, uid, "a.png", probs(0.5, 0.4, 0.05, 0.05), "m1")
    q = add(conn, uid, "b.png", probs(0.5, 0.4, 0.05, 0.05), "m2")
    assert db.review(conn, p, uid, "glioma", "ok")
    assert db.review(conn, q, uid, "meningioma")
    st = {r["id"]: r["status"] for r in db.history(conn, uid)}
    assert st == {p: "confirmed", q: "corrected"}
    assert not db.review(conn, p, uid, "glioma")  # already reviewed
    with pytest.raises(ValueError):
        db.review(conn, p, uid, "not-a-class")


def test_users_cannot_touch_each_others_predictions(conn):
    a, b = user(conn, "alice"), user(conn, "bobby")
    p = add(conn, a, "a.png", probs(0.5, 0.4, 0.05, 0.05), "m1")
    assert not db.review(conn, p, b, "glioma")
    assert not db.delete_prediction(conn, p, b)
    assert db.history(conn, b) == []


def test_history_filters_and_escapes_search_text(conn):
    uid = user(conn)
    add(conn, uid, "scan_10%.png", probs(0.9, 0.05, 0.03, 0.02), "m1")
    add(conn, uid, "scan_x.png", probs(0.05, 0.9, 0.03, 0.02), "m2")
    assert len(db.history(conn, uid, query="10%")) == 1  # % is a literal, not a wildcard
    assert len(db.history(conn, uid, cls="meningioma")) == 1
    assert len(db.history(conn, uid, status="confirmed")) == 0


def test_exact_duplicate_is_found(conn):
    uid = user(conn)
    add(conn, uid, "a.png", probs(0.9, 0.05, 0.03, 0.02), "same")
    assert db.find_duplicate(conn, uid, "same") is not None
    assert db.find_duplicate(conn, uid, "other") is None


def reviewed(conn, uid, n, md5_prefix="r"):
    for i in range(n):
        pid = add(conn, uid, "%s%d.png" % (md5_prefix, i), probs(0.5, 0.4, 0.05, 0.05), "%s%d" % (md5_prefix, i))
        db.review(conn, pid, uid, "glioma")


def test_retrain_waits_for_enough_reviewed_images(conn, tmp_path):
    s = Settings(data_dir=str(tmp_path), retrain_min=5)
    uid = user(conn)
    reviewed(conn, uid, 4)
    assert not retrain.ready(conn, s)
    reviewed(conn, uid, 1, "z")
    assert retrain.ready(conn, s)


def test_retrain_never_uses_images_of_the_test_or_external_splits(conn, tmp_path):
    manifest = tmp_path / "m.tsv"
    manifest.write_text("path\tmd5\tclass\tgroup\tsplit\tfold\n"
                        "a\tr0\tglioma\t0\ttest\t-1\n"
                        "b\tr1\tglioma\t1\texternal\t-1\n"
                        "c\tr2\tglioma\t2\ttrainval\t1\n")
    s = Settings(data_dir=str(tmp_path), retrain_min=3, blocked_manifest=str(manifest))
    uid = user(conn)
    reviewed(conn, uid, 5)  # md5 r0..r4; r0 (test) and r1 (external) are blocked, r2 (trainval) is not
    rows, n_blocked = retrain.select_training_rows(conn, s)
    assert n_blocked == 2
    assert {r["image_md5"] for r in rows} == {"r2", "r3", "r4"}


def test_round_claims_images_and_a_failed_round_releases_them(conn, tmp_path):
    s = Settings(data_dir=str(tmp_path), retrain_min=3)
    uid = user(conn)
    reviewed(conn, uid, 3)
    rid, rows = retrain.start_round(conn, s, "v1")
    assert rid and len(rows) == 3
    assert retrain.select_training_rows(conn, s)[0] == []  # claimed
    assert not db.delete_prediction(conn, rows[0]["id"], uid)  # used images stay in the history
    retrain.finish_round(conn, rid, None, ok=False)
    assert len(retrain.select_training_rows(conn, s)[0]) == 3  # released
    rid2, _ = retrain.start_round(conn, s, "v1")
    retrain.finish_round(conn, rid2, "v2", ok=True)
    assert conn.execute("SELECT status, model_to FROM retrain_rounds WHERE id = ?", (rid2,)).fetchone()[:] == ("done", "v2")


def test_start_round_refuses_when_not_enough_images(conn, tmp_path):
    s = Settings(data_dir=str(tmp_path), retrain_min=5)
    reviewed(conn, user(conn), 2)
    assert retrain.start_round(conn, s, "v1") == (None, [])
