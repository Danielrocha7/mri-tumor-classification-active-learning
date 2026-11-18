# Stage 5: external-set evaluation (Nickparvar images without a near-twin in the training data).
# Reads external_probs.npz of each run; writes per-run table, per-class results, confusion matrices, McNemar.
# Usage: python scripts/10_external_evaluation.py RUNS_DIR EXTERNAL_NPZ OUT_DIR

import sys, os, json, glob
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from scipy.stats import binomtest

runs_dir, ext_path, out_dir = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
e = np.load(ext_path, allow_pickle=True)
y, classes = e["y"], [str(c) for c in e["classes"]]
internal_mix = np.array([754, 735, 678, 822]) / 2989  # class mix of the internal test set
rng = np.random.default_rng(0)

runs = {}
for p in sorted(glob.glob(os.path.join(runs_dir, "*", "external_probs.npz"))):
    n = os.path.basename(os.path.dirname(p))
    runs[n] = dict(pred=np.load(p)["prob"].argmax(1), meta=json.load(open(os.path.join(os.path.dirname(p), "metrics.json"))))
models = sorted({n.rsplit("_s", 1)[0] for n in runs})


def boot(pred, n=2000):
    a, f = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        a.append((pred[i] == y[i]).mean())
        f.append(f1_score(y[i], pred[i], average="macro"))
    return np.percentile(a, [2.5, 97.5]), np.percentile(f, [2.5, 97.5])


def recall(pred):
    return np.array([(pred[y == k] == k).mean() for k in range(4)])


rows = []
for n, r in runs.items():
    pred = r["pred"]
    acc, f1 = (pred == y).mean(), f1_score(y, pred, average="macro")
    (a0, a1), (f0, f1_) = boot(pred)
    rec = recall(pred)
    r.update(acc=acc, f1=f1, rec=rec)
    rows.append((n, r["meta"]["test"]["macro_f1"], f1, f0, f1_, acc, a0, a1, float((rec * internal_mix).sum()), *rec))
with open(os.path.join(out_dir, "external_summary.csv"), "w") as f:
    f.write("run,internal_test_macro_f1,ext_macro_f1,f1_ci_lo,f1_ci_hi,ext_acc,acc_ci_lo,acc_ci_hi,ext_acc_reweighted_to_internal_mix," + ",".join("recall_" + c for c in classes) + "\n")
    for r in rows:
        f.write(",".join(r[0:1]) + "," + ",".join(f"{float(v):.5f}" for v in r[1:]) + "\n")

with open(os.path.join(out_dir, "external_report.txt"), "w") as f:
    for m in models:
        rs = [runs[n] for n in runs if n.rsplit("_s", 1)[0] == m]
        f.write(f"{m}: external macro-F1 mean {np.mean([r['f1'] for r in rs]):.4f} sd {np.std([r['f1'] for r in rs], ddof=1):.4f}; "
                f"acc mean {np.mean([r['acc'] for r in rs]):.4f}; reweighted acc mean {np.mean([(r['rec'] * internal_mix).sum() for r in rs]):.4f}; "
                f"internal macro-F1 mean {np.mean([r['meta']['test']['macro_f1'] for r in rs]):.4f}\n")
        f.write("  recall per class (mean over seeds): " + "; ".join(f"{c} {np.mean([r['rec'][k] for r in rs]):.3f}" for k, c in enumerate(classes)) + "\n")
        pp = np.concatenate([r["pred"] for r in rs]); yy = np.tile(y, len(rs))
        pr, rc, f1c, _ = precision_recall_fscore_support(yy, pp)
        f.write("  precision per class (pooled): " + "; ".join(f"{c} {v:.3f}" for c, v in zip(classes, pr)) + "\n")
        f.write("  confusion matrix pooled over seeds (rows true, cols pred; order " + ", ".join(classes) + ")\n" + "\n".join("   " + str(row.tolist()) for row in confusion_matrix(yy, pp)) + "\n")
    f.write("\nMcNemar exact (same seed; b = only A correct, c = only B correct)\n")
    if len(models) == 2:
        A, B = models
        for s in sorted({n.rsplit("_s", 1)[1] for n in runs}):
            ca, cb = runs[f"{A}_s{s}"]["pred"] == y, runs[f"{B}_s{s}"]["pred"] == y
            b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
            f.write(f"  seed {s}: {A} vs {B}: b={b} c={c} p={binomtest(b, b + c, 0.5).pvalue if b + c else 1.0:.2e}\n")
print(open(os.path.join(out_dir, "external_report.txt")).read())
print(open(os.path.join(out_dir, "external_summary.csv")).read())
