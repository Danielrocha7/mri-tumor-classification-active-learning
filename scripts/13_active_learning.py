"""Stage 13: simulated active learning on the leak-free pool (plan in docs/active_learning_design.md).
One run = one arm and one seed. Arms: random, entropy, margin, mcdropout, coreset, badge (ConvNeXt-Tiny with a
dropout layer, retrained from the pretrained weights every round), original (replica of the original loop with
EfficientNet-B0) and full (whole pool, reference). The test fold and the external set are only predicted, at a few budgets.
Usage (Kaggle, GPU): python scripts/13_active_learning.py --data kaggle224_seed42.npz --arm entropy --seed 42 --out results_al [--external ext.npz]
"""
import argparse
import json
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils import set_seed

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--arm", required=True, choices=["random", "entropy", "margin", "mcdropout", "coreset", "badge", "original", "full"])
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", required=True)
p.add_argument("--external", default=None, help="packed external set; predicted at the saved budgets")
p.add_argument("--quick", action="store_true", help="smoke test: tiny pool, few steps, no pretrained weights")
a = p.parse_args()

# settings fixed in the plan
N_VAL, N_START, STEPS, EVAL_EVERY, BATCH = 400, 60, 400, 40, 32
BUDGETS = list(range(60, 601, 60)) + list(range(720, 1201, 120))
SAVE_AT = {60, 300, 600, 1200}
MC_PASSES, FULL_EPOCHS = 20, 6
ORIG_START_FRAC, ORIG_INIT_EPOCHS, ORIG_ROUNDS, ORIG_QUERY, ORIG_EPOCHS = 0.01, 5, 21, 20, 3
pretrained = "DEFAULT"
if a.quick:
    N_VAL, N_START, STEPS, EVAL_EVERY, BUDGETS, SAVE_AT, MC_PASSES, FULL_EPOCHS = 40, 8, 10, 5, [8, 16, 24], {8, 24}, 3, 1
    ORIG_ROUNDS, ORIG_INIT_EPOCHS, ORIG_EPOCHS, pretrained = 2, 1, 1, None

set_seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name = "%s_s%d" % (a.arm, a.seed)
out = os.path.join(a.out, name)
os.makedirs(out, exist_ok=True)
if os.path.exists(os.path.join(out, "done.json")):
    print(name, "already done")
    sys.exit(0)

d = np.load(a.data, allow_pickle=True)
y_np, split, fold = d["y"], d["split"], d["fold"]
pool_all = np.where((split == "trainval") & (fold != 0))[0]
val_all = np.where((split == "trainval") & (fold == 0))[0]
test_idx = np.where(split == "test")[0]
if a.quick:
    r0 = np.random.default_rng(0)  # random subsets, so that all four classes are present
    pool_all, val_all, test_idx = r0.choice(pool_all, 400, replace=False), r0.choice(val_all, 100, replace=False), r0.choice(test_idx, 50, replace=False)
# validation subset and first labelled set depend on the seed only, so every arm of a seed starts from the same images
val = np.random.default_rng(a.seed * 1000 + 1).choice(val_all, N_VAL, replace=False)
x_gpu, y_gpu = torch.from_numpy(d["images"]).to(dev), torch.from_numpy(y_np).to(dev)
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
ext = None
if a.external:
    e = np.load(a.external, allow_pickle=True)
    ext = (torch.from_numpy(e["images"]).to(dev), e["y"])


def prep(idx, train, src=None):
    # same augmentation as scripts/07_train_supervised.py: flip, rotation up to 10 degrees, brightness and contrast
    x = (x_gpu if src is None else src)[idx].float().div(255).unsqueeze(1)
    if train:
        n = x.shape[0]
        flip = torch.rand(n, device=dev) < 0.5
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
        ang = (torch.rand(n, device=dev) * 20 - 10) * math.pi / 180
        c, s = torch.cos(ang), torch.sin(ang)
        theta = torch.stack([torch.stack([c, -s, torch.zeros_like(c)], 1), torch.stack([s, c, torch.zeros_like(c)], 1)], 1)
        x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False, padding_mode="zeros")
        b = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 0.4 - 0.2)
        k = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 0.4 - 0.2)
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (((x - m) * k + m) * b).clamp(0, 1)
    return (x.repeat(1, 3, 1, 1) - mean) / std


def build(kind):
    if kind == "convnext":
        # ConvNeXt-Tiny with a dropout layer before the last linear layer (the only change; needed by MC-dropout)
        m = models.convnext_tiny(weights=pretrained)
        m.classifier[2] = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[2].in_features, 4))
    else:
        # EfficientNet-B0 of the original project: partly frozen backbone and its own head
        m = models.efficientnet_b0(weights=pretrained)
        for q in list(m.features.parameters())[:100]:
            q.requires_grad = False
        n_in = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(n_in, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3), nn.Linear(256, 4))
    return m.to(dev)


def features(model, idx):
    # penultimate features of the ConvNeXt (768 values) of the images in idx, evaluation mode, no augmentation
    model.eval()
    out_f = []
    with torch.no_grad():
        for s in range(0, len(idx), 256):
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                z = model.avgpool(model.features(prep(idx[s:s + 256], False)))
                out_f.append(model.classifier[1](model.classifier[0](z)).float())
    return torch.cat(out_f)


@torch.no_grad()
def predict(model, idx, src=None):
    model.eval()
    out_p = []
    for s in range(0, len(idx), 256):
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            out_p.append(F.softmax(model(prep(idx[s:s + 256], False, src)).float(), 1).cpu())
    return torch.cat(out_p).numpy()


def val_metrics(model):
    pr = predict(model, val).argmax(1)
    t = y_np[val]
    return {"val_acc": float((pr == t).mean()), "val_macro_f1": float(f1_score(t, pr, average="macro", labels=[0, 1, 2, 3])),
            "val_recall": [float((pr[t == k] == k).mean()) if (t == k).any() else None for k in range(4)]}


def train_steps(model, labelled, steps, lr, opt=None, sched=None):
    # AdamW with cosine decay over `steps` optimisation steps of batches of 32 drawn from the labelled images;
    # the validation subset is scored every EVAL_EVERY steps and the best weights are kept
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    best, best_state, order, pos = -1.0, None, np.random.permutation(labelled), 0
    for step in range(1, steps + 1):
        model.train()
        if pos + BATCH > len(order):
            order, pos = np.random.permutation(labelled), 0
        idx = order[pos:pos + BATCH]
        pos += BATCH
        if len(idx) < 2:  # BatchNorm1d of the EfficientNet head needs at least 2 samples
            continue
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            loss = F.cross_entropy(model(prep(idx, True)), y_gpu[idx])
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % EVAL_EVERY == 0 or step == steps:
            m = val_metrics(model)
            if m["val_macro_f1"] > best:
                best, best_state = m["val_macro_f1"], {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def select(model, labelled, k, rng):
    # indices of k images of the pool that the strategy wants labelled next
    pool = np.setdiff1d(pool_all, labelled)
    if a.arm == "random":
        return rng.choice(pool, k, replace=False)
    pr = torch.from_numpy(predict(model, pool))
    if a.arm == "entropy":
        score = -(pr * pr.clamp_min(1e-12).log()).sum(1)
        return pool[score.topk(k).indices.numpy()]
    if a.arm == "margin":
        t2 = pr.topk(2, dim=1).values
        return pool[(t2[:, 0] - t2[:, 1]).topk(k, largest=False).indices.numpy()]
    f = features(model, pool)
    if a.arm == "mcdropout":
        # BALD: entropy of the mean prediction minus the mean entropy of the stochastic predictions (dropout only in the head)
        model.classifier[2].train()
        with torch.no_grad():
            ps = torch.stack([F.softmax(model.classifier[2](f), 1) for _ in range(MC_PASSES)])
        model.classifier[2].eval()
        h = lambda q: -(q * q.clamp_min(1e-12).log()).sum(-1)
        return pool[(h(ps.mean(0)) - h(ps).mean(0)).topk(k).indices.cpu().numpy()]
    if a.arm == "coreset":
        # k-center greedy: repeatedly take the image farthest from everything already labelled or chosen
        dist = torch.cdist(f, features(model, labelled)).min(1).values
        chosen = []
        for _ in range(k):
            i = int(dist.argmax())
            chosen.append(i)
            dist = torch.minimum(dist, (f - f[i]).norm(dim=1))
        return pool[np.array(chosen)]
    if a.arm == "badge":
        # gradient embedding of the last layer, (p - one-hot of the predicted class) x feature, then k-means++ seeding
        pr = pr.to(dev)
        g = ((pr - F.one_hot(pr.argmax(1), 4).float()).unsqueeze(2) * f.unsqueeze(1)).flatten(1)
        sq = (g ** 2).sum(1)
        i = int(sq.argmax())
        chosen, dist = [i], (sq + sq[i] - 2 * g @ g[i]).clamp_min(0)
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        for _ in range(k - 1):
            dist[chosen] = 0
            i = int(torch.multinomial(dist / dist.sum(), 1, generator=gen))
            chosen.append(i)
            dist = torch.minimum(dist, (sq + sq[i] - 2 * g @ g[i]).clamp_min(0))
        return pool[np.array(chosen)]


def save_probs(model, n):
    # test (and external) probabilities, saved for the final evaluation; not looked at while the plan can still change
    np.savez(os.path.join(out, "test_probs_%d.npz" % n), idx=test_idx, prob=predict(model, test_idx))
    if ext is not None:
        np.savez(os.path.join(out, "external_probs_%d.npz" % n), idx=np.arange(len(ext[1])), prob=predict(model, np.arange(len(ext[1])), ext[0]))


def record(rounds, model, labelled, t_train, t_select=0.0):
    r = {"n_labelled": len(labelled), "train_seconds": round(t_train, 1), "select_seconds": round(t_select, 1),
         "labelled_per_class": np.bincount(y_np[labelled], minlength=4).tolist()}
    r.update(val_metrics(model))
    rounds.append(r)
    json.dump(rounds, open(os.path.join(out, "rounds.json"), "w"), indent=1)
    print(name, r, flush=True)


rounds, t_start = [], time.time()
if a.arm in ("random", "entropy", "margin", "mcdropout", "coreset", "badge"):
    labelled = np.random.default_rng(a.seed * 1000 + 2).choice(pool_all, N_START, replace=False)
    sel_rng = np.random.default_rng(a.seed * 1000 + 3)
    state = os.path.join(out, "order.npy")
    if os.path.exists(state):  # resume: the labelled images chosen so far; every round retrains from scratch, so nothing else is needed
        labelled = np.load(state)
        rounds = json.load(open(os.path.join(out, "rounds.json")))
        rounds = [r for r in rounds if r["n_labelled"] < len(labelled)]
    while True:
        t0 = time.time()
        torch.manual_seed(a.seed * 100 + len(rounds))
        model = train_steps(build("convnext"), labelled, STEPS, 3e-4)
        t_train = time.time() - t0
        n = len(labelled)
        if n in SAVE_AT:
            save_probs(model, n)
        if n >= BUDGETS[-1]:
            record(rounds, model, labelled, t_train)
            break
        t0 = time.time()
        k = BUDGETS[BUDGETS.index(n) + 1] - n
        new = select(model, labelled, k, sel_rng)
        record(rounds, model, labelled, t_train, time.time() - t0)
        labelled = np.concatenate([labelled, new])
        np.save(state, labelled)
elif a.arm == "original":
    # replica of the original loop: class-balanced seed of 1% of the pool, 5 epochs, then rounds of 20 entropy-selected images
    # with 3 epochs of continued fine-tuning (Adam, lr 1e-4); the validation subset replaces the test set and there is no early stopping
    rng = np.random.default_rng(a.seed * 1000 + 2)
    n0 = max(4, int(len(pool_all) * ORIG_START_FRAC))
    first = [rng.choice(pool_all[y_np[pool_all] == k]) for k in range(4)]
    labelled = np.array(first + list(rng.choice(np.setdiff1d(pool_all, first), n0 - 4, replace=False)))
    torch.manual_seed(a.seed)
    model = build("efficientnet")
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=1e-4)

    def epochs(n_ep):
        for _ in range(n_ep):
            model.train()
            order = np.random.permutation(labelled)
            for s in range(0, len(order), 32):
                idx = order[s:s + 32]
                if len(idx) < 2:
                    continue
                loss = F.cross_entropy(model(prep(idx, True)), y_gpu[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

    t0 = time.time()
    epochs(ORIG_INIT_EPOCHS)
    for it in range(ORIG_ROUNDS):
        record(rounds, model, labelled, time.time() - t0)
        t0 = time.time()
        pool = np.setdiff1d(pool_all, labelled)
        pr = torch.from_numpy(predict(model, pool))
        labelled = np.concatenate([labelled, pool[(-(pr * pr.clamp_min(1e-12).log()).sum(1)).topk(ORIG_QUERY).indices.numpy()]])
        epochs(ORIG_EPOCHS)
    record(rounds, model, labelled, time.time() - t0)
    save_probs(model, len(labelled))
else:
    # reference: the same model as the main arms trained on the whole pool
    labelled = pool_all
    torch.manual_seed(a.seed)
    t0 = time.time()
    model = train_steps(build("convnext"), labelled, FULL_EPOCHS * math.ceil(len(labelled) / BATCH), 3e-4)
    record(rounds, model, labelled, time.time() - t0)
    save_probs(model, len(labelled))
np.save(os.path.join(out, "order.npy"), labelled)
json.dump({"arm": a.arm, "seed": a.seed, "seconds": round(time.time() - t_start, 1)}, open(os.path.join(out, "done.json"), "w"))
print(name, "done in", round(time.time() - t_start), "s")
