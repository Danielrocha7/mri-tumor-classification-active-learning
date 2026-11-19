"""Stage 15: write labels.npz (labels of the Kaggle manifest, of the external manifest and the class names).
The class index is the sorted class name, as in scripts/06_pack_images.py; rows keep the manifest order.
Usage: python scripts/15_make_labels.py splits/kaggle_splits_seed42.tsv splits/external_nickparvar_c095_manifest.tsv OUT.npz
"""
import sys
import numpy as np


def read(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path)]
    head = rows[0]
    return [dict(zip(head, r)) for r in rows[1:]]


kaggle, external, out = sys.argv[1:4]
k, e = read(kaggle), read(external)
classes = sorted({r["class"] for r in k})
assert {r["class"] for r in e} <= set(classes), "external classes differ from the training classes"
np.savez(out, y=np.array([classes.index(r["class"]) for r in k], dtype=np.int64),
         ext_y=np.array([classes.index(r["class"]) for r in e], dtype=np.int64),
         classes=np.array(classes))
print("wrote", out, len(k), len(e), classes)
