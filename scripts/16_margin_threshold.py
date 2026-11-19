"""Stage 16: how the margin (top-1 minus top-2 probability) relates to errors, to set the app's review threshold.

The threshold is chosen from VALIDATION probabilities only (fold 0, the fold that picked the epoch); the test fold
is not read. The external set is printed only as an independent check and does not choose anything.
Usage: python scripts/16_margin_threshold.py splits/kaggle_splits_seed42.tsv splits/external_nickparvar_c095_manifest.tsv results/final_runs
"""
import sys
import numpy as np

manifest, ext_manifest, runs = sys.argv[1:4]
classes = ["glioma", "meningioma", "notumor", "pituitary"]


def labels(path):
    return np.array([classes.index(l.rstrip("\n").split("\t")[2]) for l in list(open(path))[1:]])


def margin(p):
    t = np.sort(p, 1)
    return t[:, -1] - t[:, -2]


y, ye = labels(manifest), labels(ext_manifest)
grid = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.95)
for seed in (42, 43, 44):
    d = np.load(f"{runs}/convnext_tiny_s{seed}/val_probs.npz")
    m, err = margin(d["prob"]), d["prob"].argmax(1) != y[d["idx"]]
    print(f"seed {seed} validation: {len(m)} images, {err.sum()} errors, median margin of the errors {np.median(m[err]):.3f}")
    for t in grid:
        fl = m < t
        print(f"  margin < {t:<4}: flagged {100 * fl.mean():5.2f} %, errors caught {(fl & err).sum()}/{err.sum()}")
d = np.load(f"{runs}/convnext_tiny_s42/external_probs.npz")
m, err = margin(d["prob"]), d["prob"].argmax(1) != ye[d["idx"]]
print(f"seed 42 external (check only): {len(m)} images, {err.sum()} errors, median margin of the errors {np.median(m[err]):.3f}")
for t in grid:
    fl = m < t
    print(f"  margin < {t:<4}: flagged {100 * fl.mean():5.2f} %, errors caught {(fl & err).sum()}/{err.sum()}")
