"""Stage 3: overlap between a candidate external set and the training data.
For each external image, nearest dHash distance to the training images; the
external image is kept only if that distance is > THRESHOLD.
Usage (from data/interim): python ../../scripts/03_external_overlap.py TRAIN.tsv EXTERNAL.tsv THRESHOLD OUT.tsv
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils import popcount, read_hashes

train, ext, thr, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
_, _, th, tcls = read_hashes(train)
paths, _, eh, ecls = read_hashes(ext)

nearest = np.array([popcount(th ^ h).min() for h in eh])
with open(out, "w") as fo:
    fo.write("path\tclass\tnearest_distance\tkeep\n")
    for p, c, d in zip(paths, ecls, nearest):
        fo.write("%s\t%s\t%d\t%d\n" % (p, c, d, int(d > thr)))
print("external images", len(paths), "kept (distance >", thr, "):", int((nearest > thr).sum()))
