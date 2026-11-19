# Stage 14: analysis of the active-learning runs (script 13).
# Learning curves on the validation subset, then test and external results at the saved budgets,
# always as differences to random sampling with the same seed (same initial set and validation subset).
# Usage: python scripts/14_active_learning_analysis.py AL_DIR LABELS_NPZ OUT_DIR
# LABELS_NPZ holds y (all packed images), ext_y (external set) and classes; it is written on Kaggle from the packed data.

import sys, os, json, glob, logging
import numpy as np
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_dir, labels_path, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
lab = np.load(labels_path, allow_pickle=True)
y_all, ext_y = lab["y"], lab["ext_y"]

# fixed colour per arm in every figure (Okabe-Ito); the original arm keeps the EfficientNet colour of the earlier figures
ARMS = ["random", "entropy", "margin", "mcdropout", "coreset", "badge", "original", "full"]
colour = {"random": "#000000", "entropy": "#E69F00", "margin": "#009E73", "mcdropout": "#CC79A7",
          "coreset": "#0072B2", "badge": "#D55E00", "original": "#56B4E9"}
name = {"random": "Random", "entropy": "Entropy", "margin": "Margin", "mcdropout": "MC-dropout",
        "coreset": "Core-Set", "badge": "BADGE", "original": "Original"}

# runs[arm][seed] = dict(rounds, test/ext = {n_labelled: macro-F1}, test_rec/ext_rec = {n_labelled: recall per class})
runs = {}
for d in sorted(glob.glob(os.path.join(al_dir, "*_s*"))):
    arm, seed = os.path.basename(d).rsplit("_s", 1)
    if not os.path.exists(os.path.join(d, "done.json")):
        print("skipping unfinished run", d)
        continue
    r = dict(rounds=json.load(open(os.path.join(d, "rounds.json"))), test={}, ext={}, test_rec={}, ext_rec={})
    for kind in ("test", "external"):
        key = "test" if kind == "test" else "ext"
        for p in sorted(glob.glob(os.path.join(d, kind + "_probs_*.npz"))):
            n = int(os.path.basename(p).split("_")[-1][:-4])
            t = np.load(p)
            y = y_all[t["idx"]] if kind == "test" else ext_y
            pred = t["prob"].argmax(1)
            r[key][n] = f1_score(y, pred, average="macro")
            r[key + "_rec"][n] = [float((pred[y == k] == k).mean()) for k in range(4)]
    runs.setdefault(arm, {})[int(seed)] = r
print("arms:", {a: sorted(s) for a, s in runs.items()})

_trapz = getattr(np, "trapezoid", None) or np.trapz


def curve(arm, seed, field="val_macro_f1"):
    rs = runs[arm][seed]["rounds"]
    return np.array([r["n_labelled"] for r in rs]), np.array([r[field] for r in rs])


def auc(arm, seed):
    # area under the validation macro-F1 curve divided by the budget range = mean macro-F1 over the budget
    n, f = curve(arm, seed)
    return float(_trapz(f, n) / (n[-1] - n[0]))


def end_labels(axis, items, gap):
    # items: (y, text, colour); spread the labels so that they do not overlap at the right edge of the curves
    items = sorted(items)
    ys = [i[0] for i in items]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + gap)
    shift = (np.mean([i[0] for i in items]) - np.mean(ys))
    for (y0, t, c), y in zip(items, ys):
        axis.text(1215, y + shift, t, color=c, fontsize=7, va="center")


strategies = [a for a in ARMS[:6] if a in runs]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # Arial is not installed everywhere
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
lab0, lab1 = [], []
for a in strategies + (["original"] if "original" in runs else []):
    ss = sorted(runs[a])
    ns = curve(a, ss[0])[0]
    ok = [s for s in ss if len(curve(a, s)[0]) == len(ns)]
    f = np.array([curve(a, s)[1] for s in ok])
    ax[0].plot(ns, f.mean(0), "--" if a == "original" else "-", color=colour[a], lw=1.6)
    ax[0].fill_between(ns, f.min(0), f.max(0), color=colour[a], alpha=0.12, lw=0)
    if a == "original":
        ax[0].text(ns[-1] + 10, f.mean(0)[-1], name[a], color=colour[a], fontsize=7, va="center")
    else:
        lab0.append((f.mean(0)[-1], name[a], colour[a]))
    if a not in ("original", "random") and "random" in runs:
        d = np.array([curve(a, s)[1] - curve("random", s)[1] for s in ok if s in runs["random"]])
        if len(d):
            ax[1].plot(ns, d.mean(0), color=colour[a], lw=1.6)
            lab1.append((d.mean(0)[-1], name[a], colour[a]))
end_labels(ax[0], lab0, 0.017)
end_labels(ax[1], lab1, 0.0035)
ax[0].set_xlabel("Labelled images"); ax[0].set_ylabel("Validation macro-F1")
ax[0].set_title("(a) Learning curves (mean, range of seeds)", fontsize=9, loc="left")
ax[1].axhline(0, color="#999999", lw=0.8)
ax[1].set_xlabel("Labelled images"); ax[1].set_ylabel("Difference to random (macro-F1)")
ax[1].set_title("(b) Paired with random, same seed", fontsize=9, loc="left")
# (c) area under the curve per seed, as points, to show the spread that three seeds leave
for i, a in enumerate(strategies):
    v = [auc(a, s) for s in sorted(runs[a])]
    ax[2].scatter([i] * len(v), v, color=colour[a], s=22, zorder=3)
    ax[2].plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=colour[a], lw=1.6)
ax[2].set_xticks(range(len(strategies)))
ax[2].set_xticklabels([name[a] for a in strategies], rotation=35, ha="right", fontsize=8)
ax[2].set_ylabel("Mean validation macro-F1 over the budget")
ax[2].set_title("(c) Area under the curve, per seed", fontsize=9, loc="left")
for x in ax:
    x.spines[["top", "right"]].set_visible(False)
    x.tick_params(labelsize=8)
    x.xaxis.label.set_size(8); x.yaxis.label.set_size(8)
ax[0].set_xlim(0, 1450); ax[1].set_xlim(0, 1450)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(os.path.join(out, "al_curves." + ext), dpi=300)

# table: test and external macro-F1 at the saved budgets, with the per-seed difference to random
with open(os.path.join(out, "al_summary.csv"), "w") as f:
    f.write("arm,n_labelled,set,mean_macro_f1,sd,mean_diff_to_random,seeds_better_than_random,n_seeds,recall_glioma,recall_meningioma,recall_no_tumour,recall_pituitary\n")
    for a in [x for x in ARMS if x in runs and x != "full"]:
        ss = sorted(runs[a])
        for st in ("test", "ext"):
            for n in sorted(set.intersection(*[set(runs[a][s][st]) for s in ss])):
                v = np.array([runs[a][s][st][n] for s in ss])
                rec = np.mean([runs[a][s][st + "_rec"][n] for s in ss], axis=0)
                if a != "random" and "random" in runs and all(n in runs["random"].get(s, {}).get(st, {}) for s in ss):
                    dv = np.array([runs[a][s][st][n] - runs["random"][s][st][n] for s in ss])
                    dd, nb = f"{dv.mean():.4f}", str(int((dv > 0).sum()))
                else:
                    dd, nb = "", ""
                f.write(f"{a},{n},{'test' if st == 'test' else 'external'},{v.mean():.4f},{v.std(ddof=1) if len(v) > 1 else 0:.4f},{dd},{nb},{len(ss)},"
                        + ",".join(f"{x:.4f}" for x in rec) + "\n")

# full-pool reference
if "full" in runs:
    ss = [s for s in runs["full"] if runs["full"][s]["test"]]
    v = [runs["full"][s]["test"][max(runs["full"][s]["test"])] for s in ss]
    e = [runs["full"][s]["ext"][max(runs["full"][s]["ext"])] for s in ss]
    open(os.path.join(out, "al_full_reference.txt"), "w").write(f"full pool: test macro-F1 {np.mean(v):.4f}, external macro-F1 {np.mean(e):.4f}, seeds {len(v)}\n")
print("done")
