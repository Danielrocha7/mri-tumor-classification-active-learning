"""Retraining round: versions, leakage guard and forgetting check. Uses a tiny real fine-tuning step on random weights."""
import os
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
try:
    import torch
except Exception:  # torch missing or broken in this environment
    torch = None
pytestmark = pytest.mark.skipif(torch is None, reason="torch not available")

from app.backend import auth, db, trainer
from app.backend.model_service import build_model
from app.backend.settings import Settings

CLS = db.CLASSES


def make(tmp_path, n, retrain_min=3, manifest=""):
    s = Settings(data_dir=str(tmp_path / "data"), retrain_min=retrain_min, blocked_manifest=manifest, ft_epochs=1)
    s.ensure_dirs()
    torch.manual_seed(0)
    torch.save(build_model().state_dict(), os.path.join(s.weights_dir, "v1.pt"))
    c = db.connect(s.db_path)
    auth.register(c, "alice", "correct horse")
    uid = auth.login(c, "alice", "correct horse")
    ids = []
    for i in range(n):
        p = str(tmp_path / ("i%d.png" % i))
        Image.fromarray(np.random.RandomState(i).randint(0, 255, (64, 64), dtype=np.uint8)).save(p)
        pr = {k: 0.25 for k in CLS}
        pid = db.add_prediction(c, uid, p, "md5-%d" % i, pr, "v1", True)
        db.review(c, pid, uid, CLS[i % 4], "")
        ids.append(pid)
    return s, c, ids


def test_round_creates_next_version_and_marks_rows(tmp_path):
    s, c, ids = make(tmp_path, 4)
    out = trainer.run_round(c, s)
    assert out["status"] == "promoted" and out["version"] == "v2" and out["n_train"] == 4
    assert [n for n, _ in trainer.versions(s)] == [1, 2]
    assert c.execute("SELECT COUNT(*) FROM predictions WHERE round_id = ?", (out["round_id"],)).fetchone()[0] == 4
    assert trainer.run_round(c, s) is None  # nothing left to train on


def test_blocked_images_are_never_loaded_for_training(tmp_path):
    man = tmp_path / "m.tsv"
    man.write_text("path\tmd5\tclass\tgroup\tsplit\tfold\nx\tmd5-0\tglioma\t0\ttest\t-1\n")
    s, c, ids = make(tmp_path, 4, manifest=str(man))
    seen = []

    def spy(w_in, w_out, items, **kw):
        seen.extend(p for p, _ in items)
        return trainer.fine_tune(w_in, w_out, items, **kw)

    trainer.run_round(c, s, train_fn=spy)
    assert len(seen) == 3 and not any(p.endswith("i0.png") for p in seen)


def test_failed_training_releases_rows_and_leaves_no_file(tmp_path):
    s, c, ids = make(tmp_path, 3)

    def boom(*a, **k):
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        trainer.run_round(c, s, train_fn=boom)
    assert c.execute("SELECT COUNT(*) FROM predictions WHERE round_id IS NOT NULL").fetchone()[0] == 0
    assert [n for n, _ in trainer.versions(s)] == [1]


def test_forgetting_check_rejects_a_worse_model(tmp_path):
    s, c, ids = make(tmp_path, 3)
    assert trainer.run_round(c, s)["status"] == "promoted"
    # second round: the old model scores some accuracy on round-1 images; a training function that wrecks the weights is rejected
    uid = c.execute("SELECT id FROM users").fetchone()[0]
    for i in range(10, 13):
        p = str(tmp_path / ("i%d.png" % i))
        Image.fromarray(np.random.RandomState(i).randint(0, 255, (64, 64), dtype=np.uint8)).save(p)
        pid = db.add_prediction(c, uid, p, "md5-%d" % i, {k: 0.25 for k in CLS}, "v2", True)
        db.review(c, pid, uid, "glioma", "")
    # make round-1 images match what v2 predicts, so that accuracy before is 1.0
    from app.backend.model_service import ModelService, load_image
    svc = ModelService(trainer.latest(s)[1])
    for r in c.execute("SELECT * FROM predictions WHERE round_id IS NOT NULL").fetchall():
        lab = CLS[int(svc.predict_arrays([load_image(r["image_path"])])[0].argmax())]
        c.execute("UPDATE predictions SET true_class = ? WHERE id = ?", (lab, r["id"]))
    c.commit()

    def wreck(w_in, w_out, items, **kw):
        m = build_model()
        torch.manual_seed(123)
        for q in m.parameters():
            torch.nn.init.normal_(q, std=0.5) if q.dim() > 1 else None
        torch.save(m.state_dict(), w_out)
        return {"n": len(items), "loss": 9.9}

    out = trainer.run_round(c, s, train_fn=wreck)
    assert out["prior_before"] == 1.0
    assert out["status"] == "rejected" and [n for n, _ in trainer.versions(s)] == [1, 2]
