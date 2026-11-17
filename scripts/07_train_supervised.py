"""Stage 7: fully supervised baseline on one cross-validation fold.
Trains on the other folds, picks the epoch by validation macro-F1, and writes
validation probabilities. Test probabilities are written only with --predict-test.
Usage (Colab, GPU): python scripts/07_train_supervised.py --data kaggle224_seed42.npz --model resnet50 --val-fold 0 --out results/resnet50_f0
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
p.add_argument("--model", required=True, choices=["efficientnet_b0", "resnet50", "convnext_tiny", "dinov2_linear"])
p.add_argument("--val-fold", type=int, default=0)
p.add_argument("--epochs", type=int, default=12)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--weight-decay", type=float, default=1e-4)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", required=True)
p.add_argument("--predict-test", action="store_true")
p.add_argument("--quick", action="store_true", help="smoke test: 256 images, 1 epoch")
p.add_argument("--no-pretrained", action="store_true")
a = p.parse_args()

set_seed(a.seed)
torch.manual_seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(a.out, exist_ok=True)

d = np.load(a.data)
x_all, y_all, split, fold = d["images"], d["y"], d["split"], d["fold"]
tr = np.where((split == "trainval") & (fold != a.val_fold))[0]
va = np.where((split == "trainval") & (fold == a.val_fold))[0]
te = np.where(split == "test")[0]
if a.quick:
    tr, va, te, a.epochs = tr[:256], va[:128], te[:128], 1
x_gpu = torch.from_numpy(x_all).to(dev)
y_gpu = torch.from_numpy(y_all).to(dev)
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def prep(idx, train):
    x = x_gpu[idx].float().div(255).unsqueeze(1)
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
        x = ((x - m) * k + m) * b
        x = x.clamp(0, 1)
    return (x.repeat(1, 3, 1, 1) - mean) / std


def build(name):
    w = None if a.no_pretrained else "DEFAULT"
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
    elif name == "resnet50":
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, 4)
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=w)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 4)
    else:
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=not a.no_pretrained)
        for q in backbone.parameters():
            q.requires_grad = False
        m = nn.Sequential(backbone, nn.Linear(768, 4))
    return m


model = build(a.model).to(dev)
params = [q for q in model.parameters() if q.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
steps = a.epochs * math.ceil(len(tr) / a.batch)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")


@torch.no_grad()
def predict(idx):
    model.eval()
    out = []
    for s in range(0, len(idx), 128):
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            out.append(F.softmax(model(prep(idx[s:s + 128], False)).float(), 1).cpu())
    return torch.cat(out).numpy()


def metrics(prob, idx):
    pred, true = prob.argmax(1), y_all[idx]
    return {"accuracy": float((pred == true).mean()), "macro_f1": float(f1_score(true, pred, average="macro"))}


history, best, best_state = [], -1.0, None
for ep in range(a.epochs):
    model.train()
    t0 = time.time()
    perm = np.random.permutation(tr)
    loss_sum = 0.0
    for s in range(0, len(perm), a.batch):
        idx = perm[s:s + a.batch]
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            loss = F.cross_entropy(model(prep(idx, True)), y_gpu[idx])
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        loss_sum += loss.item() * len(idx)
    m = metrics(predict(va), va)
    m.update(epoch=ep + 1, loss=loss_sum / len(perm), seconds=round(time.time() - t0, 1))
    history.append(m)
    print(m, flush=True)
    if m["macro_f1"] > best:
        best, best_state = m["macro_f1"], {k: v.detach().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
val_prob = predict(va)
result = {"args": vars(a), "history": history, "best_epoch": int(np.argmax([h["macro_f1"] for h in history])) + 1,
          "val": metrics(val_prob, va)}
np.savez(os.path.join(a.out, "val_probs.npz"), idx=va, prob=val_prob)
if a.predict_test:
    test_prob = predict(te)
    result["test"] = metrics(test_prob, te)
    np.savez(os.path.join(a.out, "test_probs.npz"), idx=te, prob=test_prob)
json.dump(result, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
print("best epoch", result["best_epoch"], "val", result["val"], "test" if a.predict_test else "(test not used)", result.get("test", ""))
