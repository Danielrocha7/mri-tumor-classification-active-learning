# Stage 6: internal test against external set, same models and seeds.
# Per-class recall with bootstrap CIs (groups for the internal test, images for the external set), table and two figures.
# Usage: python scripts/11_internal_vs_external.py RUNS_DIR DATA_NPZ EXTERNAL_NPZ OUT_DIR

import sys, os, json, glob
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs_dir, data_path, ext_path, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
d, e = np.load(data_path, allow_pickle=True), np.load(ext_path, allow_pickle=True)
classes = ["glioma", "meningioma", "no tumour", "pituitary"]
rng = np.random.default_rng(0)
colour = {"efficientnet_b0_original": "#56B4E9", "convnext_tiny": "#D55E00"}  # same as script 08
label = {"efficientnet_b0_original": "EfficientNet-B0 (original)", "convnext_tiny": "ConvNeXt-Tiny"}

runs = {}
for p in sorted(glob.glob(os.path.join(runs_dir, "*", "test_probs.npz"))):
    n, dd = os.path.basename(os.path.dirname(p)), os.path.dirname(p)
    runs[n] = dict(it=np.load(p), ex=np.load(os.path.join(dd, "external_probs.npz")), meta=json.load(open(os.path.join(dd, "metrics.json"))))
models = [m for m in colour if any(n.rsplit("_s", 1)[0] == m for n in runs)]


def recall_by_class(y, pred):
    return np.array([(pred[y == k] == k).mean() for k in range(4)])


def boot_recall(m, n=2000):
    # one resample per iteration, shared by the seeds of the model; the statistic is the mean over seeds
    rs = [runs[k] for k in runs if k.rsplit("_s", 1)[0] == m]
    idx_i = rs[0]["it"]["idx"]
    yi, gi = d["y"][idx_i], d["group"][idx_i]
    ug = np.unique(gi)
    mem = {g: np.where(gi == g)[0] for g in ug}
    ye = e["y"]
    pi = [r["it"]["prob"].argmax(1) for r in rs]
    pe = [r["ex"]["prob"].argmax(1) for r in rs]
    bi, be = [], []
    for _ in range(n):
        ii = np.concatenate([mem[g] for g in rng.choice(ug, len(ug))])
        ie = rng.integers(0, len(ye), len(ye))
        bi.append(np.mean([recall_by_class(yi[ii], p[ii]) for p in pi], 0))
        be.append(np.mean([recall_by_class(ye[ie], p[ie]) for p in pe], 0))
    pt_i = np.mean([recall_by_class(yi, p) for p in pi], 0)
    pt_e = np.mean([recall_by_class(ye, p) for p in pe], 0)
    return pt_i, np.percentile(bi, [2.5, 97.5], 0), pt_e, np.percentile(be, [2.5, 97.5], 0)


res = {m: boot_recall(m) for m in models}
with open(os.path.join(out, "internal_vs_external_recall.csv"), "w") as f:
    f.write("model,class,n_internal,n_external,internal_recall,int_ci_lo,int_ci_hi,external_recall,ext_ci_lo,ext_ci_hi,difference\n")
    for m in models:
        pi, ci, pe, ce = res[m]
        r0 = next(runs[k] for k in runs if k.rsplit("_s", 1)[0] == m)
        for k, c in enumerate(classes):
            f.write(f"{m},{c},{int((d['y'][r0['it']['idx']] == k).sum())},{int((e['y'] == k).sum())},{pi[k]:.4f},{ci[0][k]:.4f},{ci[1][k]:.4f},{pe[k]:.4f},{ce[0][k]:.4f},{ce[1][k]:.4f},{pe[k] - pi[k]:.4f}\n")

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

# figure 1: (a) macro-F1 per seed, internal against external; (b) recall per class
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.5]})
for j, m in enumerate(models):
    rs = [runs[k] for k in runs if k.rsplit("_s", 1)[0] == m]
    fi = [r["meta"]["test"]["macro_f1"] for r in rs]
    from sklearn.metrics import f1_score
    fe = [f1_score(e["y"], r["ex"]["prob"].argmax(1), average="macro") for r in rs]
    for u, v in zip(fi, fe):
        a1.plot([0, 1], [u, v], color=colour[m], lw=0.8, alpha=0.6)
    a1.scatter(np.zeros(len(fi)), fi, s=14, color=colour[m], zorder=3)
    a1.scatter(np.ones(len(fe)), fe, s=14, color=colour[m], zorder=3)
    a1.text(1.06, np.mean(fe), label[m].replace(" (original)", "\n(original)"), color=colour[m], va="center", fontsize=7)
a1.set_xlim(-0.2, 1.9)
a1.set_xticks([0, 1])
a1.set_xticklabels(["Internal test", "External set"])
a1.set_ylabel("Macro-F1 (one dot per seed)")
a1.set_ylim(0.9, 1.0)
a1.text(-0.02, 1.02, "(a)", transform=a1.transAxes, fontweight="bold")
for j, m in enumerate(models):
    pi, ci, pe, ce = res[m]
    off = (-0.14, 0.14)[j]
    for k in range(4):
        y = k + off
        a2.plot([pi[k], pe[k]], [y, y], color=colour[m], lw=0.8, alpha=0.6)
        a2.hlines(y, ce[0][k], ce[1][k], color=colour[m], lw=2, alpha=0.35)
        a2.scatter(pi[k], y, s=16, color=colour[m], zorder=3)
        a2.scatter(pe[k], y, s=16, facecolor="white", edgecolor=colour[m], zorder=3)
a2.set_yticks(range(4))
a2.set_yticklabels(classes)
a2.invert_yaxis()
a2.set_xlabel("Recall, mean of 3 seeds")
a2.text(0.805, 3.62, "filled: internal; open: external, bar = 95% CI", fontsize=6.5, color="0.3")
a2.set_xlim(0.8, 1.005)
a2.text(-0.02, 1.02, "(b)", transform=a2.transAxes, fontweight="bold")
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(os.path.join(out, "internal_vs_external." + ext), dpi=300)

# figure 2: external confusion matrices, pooled over seeds, row-normalised
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, m, tag in zip(axs, models[::-1], "ab"):
    rs = [runs[k] for k in runs if k.rsplit("_s", 1)[0] == m]
    yy = np.tile(e["y"], len(rs))
    pp = np.concatenate([r["ex"]["prob"].argmax(1) for r in rs])
    cm = confusion_matrix(yy, pp)
    nm = cm / cm.sum(1, keepdims=True)
    ax.imshow(nm, cmap="viridis", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7, color="white" if nm[i, j] < 0.6 else "black")
    ax.set_xticks(range(4))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(label[m], fontsize=8)
    ax.text(-0.02, 1.14, f"({tag})", transform=ax.transAxes, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(os.path.join(out, "external_confusion." + ext), dpi=300)
print(open(os.path.join(out, "internal_vs_external_recall.csv")).read())
