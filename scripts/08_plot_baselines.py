"""Stage 8: figure and table for the supervised baselines.
Reads every metrics.json in RESULTS_DIR (one per model and fold). Panel (a): validation
macro-F1 per epoch, mean and range over folds. Panel (b): best validation macro-F1 per
model, one dot per fold. Saves SVG and PNG plus a CSV summary in OUT_DIR.
Usage (Colab or local): python scripts/08_plot_baselines.py RESULTS_DIR OUT_DIR
"""
import csv
import glob
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

results, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

# one colour per model, kept identical in every figure (Okabe-Ito)
colour = {"efficientnet_b0_original": "#56B4E9", "efficientnet_b0": "#0072B2", "resnet50": "#E69F00",
          "convnext_tiny": "#D55E00", "dinov2_linear": "#009E73"}
label = {"efficientnet_b0_original": "EfficientNet-B0 (original)", "efficientnet_b0": "EfficientNet-B0",
         "resnet50": "ResNet-50", "convnext_tiny": "ConvNeXt-Tiny", "dinov2_linear": "DINOv2-B (linear)"}
short = {"efficientnet_b0_original": "EffNet-B0\n(original)", "efficientnet_b0": "EffNet-B0", "resnet50": "ResNet-50",
         "convnext_tiny": "ConvNeXt-T", "dinov2_linear": "DINOv2-B\n(linear)"}

runs = {}
for f in sorted(glob.glob(os.path.join(results, "*", "metrics.json"))):
    m = json.load(open(f))
    if m["args"].get("quick"):
        continue
    runs.setdefault(m["args"]["model"], []).append(m)
models = [k for k in colour if k in runs]

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})

ends = {}
for k in models:
    curves = np.array([[h["macro_f1"] for h in r["history"]] for r in runs[k]])
    ep = np.arange(1, curves.shape[1] + 1)
    ax1.plot(ep, curves.mean(0), color=colour[k], lw=1.4)
    if len(curves) > 1:
        ax1.fill_between(ep, curves.min(0), curves.max(0), color=colour[k], alpha=0.15, lw=0)
    ends[k] = curves.mean(0)[-1]
# end labels, pushed apart symmetrically so they do not overlap
ypos = dict(ends)
for _ in range(100):
    order = sorted(ypos, key=ypos.get)
    for a, b in zip(order, order[1:]):
        gap = 0.0045 - (ypos[b] - ypos[a])
        if gap > 0:
            ypos[a] -= gap / 2
            ypos[b] += gap / 2
for k in models:
    ax1.text(ep[-1] + 0.3, ypos[k], label[k], color=colour[k], va="center", fontsize=7)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Validation macro-F1")
ax1.set_ylim(0.94, 1.0)
ax1.set_xlim(1, 17)
ax1.set_xticks([1, 4, 8, 12])

rows = []
for i, k in enumerate(models):
    best = np.array([r["val"]["macro_f1"] for r in runs[k]])
    ax2.scatter(np.full(len(best), i), best, s=14, color=colour[k], zorder=3)
    ax2.hlines(best.mean(), i - 0.3, i + 0.3, color=colour[k], lw=1.6, zorder=2)
    rows.append([label[k], len(best), "%.4f" % best.mean(), "%.4f" % (best.std(ddof=1) if len(best) > 1 else 0.0)])
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels([short[k] for k in models], fontsize=6.5)
ax2.set_ylabel("Best validation macro-F1")
ax2.set_ylim(0.98, 1.0)
ax2.set_yticks(np.arange(0.98, 1.0001, 0.005))
ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
ax2.set_xlim(-0.6, len(models) - 0.4)
for ax, t in ((ax1, "(a)"), (ax2, "(b)")):
    ax.text(-0.14 if ax is ax1 else -0.22, 1.04, t, transform=ax.transAxes, fontsize=9, fontweight="bold")
fig.tight_layout()
for ext in ("svg", "png"):
    fig.savefig(os.path.join(out, "baselines_validation." + ext), dpi=300)

with open(os.path.join(out, "baselines_summary.csv"), "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["model", "folds", "mean_best_val_macro_f1", "sd"])
    w.writerows(rows)
for r in rows:
    print(r)
