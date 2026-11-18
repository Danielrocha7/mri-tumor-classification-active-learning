"""Stage 4: overlap between a candidate external set and the training data.
An external image is dropped if any training image has a pixel correlation
(64x64 thumbnails) >= MIN_CORR with it. The nearest dHash distance is reported.
Usage (from data/interim): python ../../scripts/04_external_overlap.py TRAIN.tsv TRAINTHUMBS EXT.tsv EXTTHUMBS MIN_CORR OUT.tsv
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils import load_thumbs, popcount, read_hashes

train, tpre, ext, epre, minc, out = sys.argv[1:7]
minc = float(minc)
tpaths, _, th, _ = read_hashes(train)
epaths, _, eh, ecls = read_hashes(ext)
tz = load_thumbs(tpre, len(tpaths))
ez = load_thumbs(epre, len(epaths))

# highest correlation of each external image with any training image
best = np.zeros(len(epaths), dtype=np.float32)
for s in range(0, len(epaths), 1000):
    best[s:s + 1000] = (ez[s:s + 1000] @ tz.T).max(axis=1)

# keep = 1 when no training image is similar enough to the external one
with open(out, "w") as fo:
    fo.write("path\tclass\tnearest_hash_distance\tmax_correlation\tkeep\n")
    for i in range(len(epaths)):
        fo.write("%s\t%s\t%d\t%.3f\t%d\n" % (epaths[i], ecls[i], popcount(th ^ eh[i]).min(), best[i], int(best[i] < minc)))
print("external images", len(epaths), "kept (max correlation <", minc, "):", int((best < minc).sum()))
