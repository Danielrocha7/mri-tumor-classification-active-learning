"""Retraining on CPU: a short, low-learning-rate fine-tuning of the current model on the images reviewed since the
last round, run as a background thread. The images come only from retrain.start_round, which has already dropped
everything whose md5 belongs to the held-out test or external splits.

Weights are versioned (models/v1.pt, v2.pt, ...). A new version is promoted only if accuracy on the images of
EARLIER rounds does not drop (a forgetting check). Those images were training data of the earlier rounds, so this
is not a test of generalisation: with so few reviewed images the app cannot measure that, and says so."""
import math
import os
import re
import shutil
import threading

import numpy as np

from . import retrain
from .db import CLASSES
from .model_service import ModelService, load_image, build_model, MEAN, STD


def versions(settings):
    """[(number, path)] of the weight files models/v<n>.pt, ascending."""
    out = []
    if os.path.isdir(settings.weights_dir):
        for f in os.listdir(settings.weights_dir):
            m = re.fullmatch(r"v(\d+)\.pt", f)
            if m:
                out.append((int(m.group(1)), os.path.join(settings.weights_dir, f)))
    return sorted(out)


def latest(settings):
    v = versions(settings)
    return v[-1] if v else (None, None)


def install_initial(settings, best_pt):
    """Copy the weights trained on Kaggle (best.pt) in as version 1."""
    settings.ensure_dirs()
    dst = os.path.join(settings.weights_dir, "v1.pt")
    shutil.copyfile(best_pt, dst)
    return dst


def _augment(x, torch):
    """Same augmentation as the training script: flip, rotation up to 10 degrees, brightness/contrast up to 20 %."""
    import torch.nn.functional as F
    n = x.shape[0]
    flip = torch.rand(n) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    ang = (torch.rand(n) * 20 - 10) * math.pi / 180
    c, s = torch.cos(ang), torch.sin(ang)
    z = torch.zeros_like(c)
    theta = torch.stack([torch.stack([c, -s, z], 1), torch.stack([s, c, z], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False, padding_mode="zeros")
    b = 1 + (torch.rand(n, 1, 1, 1) * 0.4 - 0.2)
    k = 1 + (torch.rand(n, 1, 1, 1) * 0.4 - 0.2)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * k + m).mul(b).clamp(0, 1)


def fine_tune(weights_in, weights_out, items, epochs=5, lr=1e-5, batch=8, seed=0, weight_decay=1e-4):
    """items: [(image_path, class_index)]. Saves the state_dict to weights_out and returns {"n", "loss"}."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = build_model()
    model.load_state_dict(torch.load(weights_in, map_location="cpu", weights_only=True))
    model.train()
    x_all = torch.from_numpy(np.stack([load_image(p) for p, _ in items])).float().div(255).unsqueeze(1)
    y_all = torch.tensor([c for _, c in items])
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * math.ceil(len(items) / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    last = 0.0
    for _ in range(epochs):
        perm = rng.permutation(len(items))
        total = 0.0
        for s in range(0, len(perm), batch):
            idx = torch.from_numpy(perm[s:s + batch])
            x = (_augment(x_all[idx], torch).repeat(1, 3, 1, 1) - mean) / std
            loss = F.cross_entropy(model(x), y_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        last = total / len(items)
    os.makedirs(os.path.dirname(weights_out), exist_ok=True)
    torch.save(model.state_dict(), weights_out)
    return {"n": len(items), "loss": last}


def _accuracy(weights, rows):
    if not rows:
        return None
    svc = ModelService(weights)
    p = svc.predict_arrays([load_image(r["image_path"]) for r in rows])
    return float(np.mean([CLASSES[i] == r["true_class"] for i, r in zip(p.argmax(1), rows)]))


def run_round(conn, settings, train_fn=fine_tune):
    """One retraining round. Returns a dict with the outcome, or None if there are not enough reviewed images.
    status 'promoted' (new version available), 'rejected' (forgetting check failed, rows released) or raises."""
    n, cur = latest(settings)
    if cur is None:
        raise FileNotFoundError("no model weights installed (models/v1.pt)")
    round_id, rows = retrain.start_round(conn, settings, "v%d" % n)
    if round_id is None:
        return None
    new = os.path.join(settings.weights_dir, "v%d.pt" % (n + 1))
    try:
        # images of earlier rounds, for the forgetting check
        prior = [r for r in conn.execute(
            "SELECT * FROM predictions WHERE round_id IS NOT NULL AND round_id < ? AND status IN ('confirmed','corrected')",
            (round_id,)).fetchall()]
        items = [(r["image_path"], CLASSES.index(r["true_class"])) for r in rows]
        stats = train_fn(cur, new, items, epochs=settings.ft_epochs, lr=settings.ft_lr)
        before, after = _accuracy(cur, prior), _accuracy(new, prior)
        out = {"round_id": round_id, "n_train": len(rows), "loss": stats["loss"], "prior_before": before, "prior_after": after}
        if before is not None and after < before - settings.max_forget_drop:
            os.remove(new)
            retrain.finish_round(conn, round_id, None, ok=False)
            out["status"] = "rejected"
            return out
        retrain.finish_round(conn, round_id, "v%d" % (n + 1), ok=True)
        out["status"] = "promoted"
        out["version"] = "v%d" % (n + 1)
        return out
    except Exception:
        if os.path.exists(new):
            os.remove(new)
        retrain.finish_round(conn, round_id, None, ok=False)
        raise


def run_round_in_background(db_path, settings, on_done=None, train_fn=fine_tune):
    """Runs a round in a daemon thread with its own connection (sqlite connections are not shared across threads)."""
    from .db import connect

    def work():
        c = connect(db_path)
        try:
            res = run_round(c, settings, train_fn)
        except Exception as e:  # reported to the UI; the round is already marked failed
            res = {"status": "failed", "error": str(e)}
        finally:
            c.close()
        if on_done:
            on_done(res)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    return t
