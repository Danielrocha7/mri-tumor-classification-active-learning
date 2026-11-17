"""Stage 5: group-aware splits.
Groups with more than one class are excluded and listed. One fold of a 7-fold
StratifiedGroupKFold is the test set (about 14 %); the rest is split into 6
folds for cross-validation. No group is ever in two partitions.
Usage (from data/interim): python ../../scripts/05_make_splits.py GROUPS.tsv OUTDIR SEED
"""
import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

groups_file, outdir, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [l.rstrip("\n").split("\t") for l in open(groups_file)]
head, rows = rows[0], rows[1:]
col = {h: k for k, h in enumerate(head)}

classes = {}
for r in rows:
    classes.setdefault(r[col["group"]], set()).add(r[col["class"]])
conflict = {g for g, c in classes.items() if len(c) > 1}
kept = [r for r in rows if r[col["group"]] not in conflict]

with open(os.path.join(outdir, "excluded_label_conflicts.tsv"), "w") as fo:
    fo.write("path\tmd5\tclass\tgroup\n")
    for r in rows:
        if r[col["group"]] in conflict:
            fo.write("%s\t%s\t%s\t%s\n" % (r[col["path"]], r[col["md5"]], r[col["class"]], r[col["group"]]))

y = np.array([r[col["class"]] for r in kept])
g = np.array([r[col["group"]] for r in kept])
split = np.array(["trainval"] * len(kept), dtype=object)
fold = np.full(len(kept), -1)

_, test_idx = next(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed).split(y, y, g))
split[test_idx] = "test"
rest = np.where(split == "trainval")[0]
for k, (_, val) in enumerate(StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed).split(rest, y[rest], g[rest])):
    fold[rest[val]] = k

with open(os.path.join(outdir, "kaggle_splits_seed%d.tsv" % seed), "w") as fo:
    fo.write("path\tmd5\tclass\tgroup\tsplit\tfold\n")
    for i, r in enumerate(kept):
        fo.write("%s\t%s\t%s\t%s\t%s\t%d\n" % (r[col["path"]], r[col["md5"]], y[i], g[i], split[i], fold[i]))

print("excluded groups", len(conflict), "images", len(rows) - len(kept), "kept", len(kept))
print("test", int((split == "test").sum()), "trainval", int((split == "trainval").sum()))
for c in sorted(set(y)):
    print(c, "test", int(((split == "test") & (y == c)).sum()), "trainval", int(((split == "trainval") & (y == c)).sum()))
