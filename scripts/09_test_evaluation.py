# Stage 5: closed test-set evaluation of the final supervised models (test opened once per model).
# Reads test_probs.npz of each run, writes a summary table, bootstrap CIs and paired McNemar tests.
# Usage: python scripts/09_test_evaluation.py RUNS_DIR DATA_NPZ OUT_DIR

import sys, os, json, glob
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from scipy.stats import binomtest

runs_dir, data_path, out_dir = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
d = np.load(data_path, allow_pickle=True)
y_all, group_all, classes = d["y"], d["group"], [str(c) for c in d["classes"]]
rng = np.random.default_rng(0)

runs = {}
for p in sorted(glob.glob(os.path.join(runs_dir, "*", "test_probs.npz"))):
    name = os.path.basename(os.path.dirname(p))
    t = np.load(p)
    runs[name] = dict(idx=t["idx"], pred=t["prob"].argmax(1), meta=json.load(open(os.path.join(os.path.dirname(p), "metrics.json"))))
models = sorted({n.rsplit("_s", 1)[0] for n in runs})


# group-level bootstrap: resample near-duplicate groups, not images, so twins do not make the CI too narrow
def boot_ci(idx, pred, n=2000):
    y, g = y_all[idx], group_all[idx]
    ug = np.unique(g)
    members = {k: np.where(g == k)[0] for k in ug}
    accs, f1s = [], []
    for _ in range(n):
        ii = np.concatenate([members[k] for k in rng.choice(ug, len(ug))])
        accs.append((pred[ii] == y[ii]).mean())
        f1s.append(f1_score(y[ii], pred[ii], average="macro"))
    return np.percentile(accs, [2.5, 97.5]), np.percentile(f1s, [2.5, 97.5])


rows = []
for n, r in runs.items():
    y = y_all[r["idx"]]
    acc, f1 = (r["pred"] == y).mean(), f1_score(y, r["pred"], average="macro")
    (a0, a1), (f0, f1_) = boot_ci(r["idx"], r["pred"])
    r.update(acc=acc, f1=f1)
    rows.append((n, r["meta"]["best_epoch"], r["meta"]["val"]["macro_f1"], acc, a0, a1, f1, f0, f1_, int((r["pred"] != y).sum()), len(y)))
with open(os.path.join(out_dir, "test_summary.csv"), "w") as f:
    f.write("run,best_epoch,val_macro_f1,test_acc,acc_ci_lo,acc_ci_hi,test_macro_f1,f1_ci_lo,f1_ci_hi,errors,n_test\n")
    for r in rows:
        f.write(",".join(str(round(float(v), 5)) if isinstance(v, (float, np.floating)) else str(v) for v in r) + "\n")

with open(os.path.join(out_dir, "test_report.txt"), "w") as f:
    for m in models:
        rs = [runs[n] for n in runs if n.rsplit("_s", 1)[0] == m]
        f.write(f"{m}: test macro-F1 mean {np.mean([r['f1'] for r in rs]):.4f} sd {np.std([r['f1'] for r in rs], ddof=1):.4f}; "
                f"acc mean {np.mean([r['acc'] for r in rs]):.4f} sd {np.std([r['acc'] for r in rs], ddof=1):.4f}; "
                f"val macro-F1 mean {np.mean([r['meta']['val']['macro_f1'] for r in rs]):.4f}\n")
        y = np.concatenate([y_all[r["idx"]] for r in rs]); p = np.concatenate([r["pred"] for r in rs])
        pr, rc, f1c, _ = precision_recall_fscore_support(y, p)
        f.write("  per-class (pooled over seeds): " + "; ".join(f"{c} P{a:.3f} R{b:.3f} F1 {e:.3f}" for c, a, b, e in zip(classes, pr, rc, f1c)) + "\n")
        f.write("  confusion matrix (rows true, cols pred, pooled over seeds)\n" + "\n".join("   " + str(row.tolist()) for row in confusion_matrix(y, p)) + "\n")
    # exact McNemar, same seed, model A vs model B on the same test images
    f.write("\nMcNemar exact (same seed, paired on test images; b = only A correct, c = only B correct)\n")
    if len(models) == 2:
        A, B = models
        for s in sorted({n.rsplit("_s", 1)[1] for n in runs}):
            ra, rb = runs[f"{A}_s{s}"], runs[f"{B}_s{s}"]
            assert (ra["idx"] == rb["idx"]).all()
            y = y_all[ra["idx"]]; ca, cb = ra["pred"] == y, rb["pred"] == y
            b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
            pv = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
            f.write(f"  seed {s}: {A} vs {B}: b={b} c={c} p={pv:.2e}\n")
print(open(os.path.join(out_dir, "test_report.txt")).read())
print(open(os.path.join(out_dir, "test_summary.csv")).read())
