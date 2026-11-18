"""Stage 2: 64x64 grayscale thumbnails, used to confirm near-duplicates by pixel correlation.
Rows follow HASHES.tsv, 1000 images per chunk file. Stops after SECONDS; run again to resume.
Usage (from data/interim): python ../../scripts/02_make_thumbnails.py ROOT HASHES.tsv OUTPREFIX SECONDS
"""
import os
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils import thumb

root, hashes, prefix, budget = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
# same order as the hash table: thumbnail i belongs to row i
paths = [l.split("\t")[0] for l in open(hashes)][1:]
start = time.time()


def work(rel):
    return thumb(open(os.path.join(root, rel), "rb").read())


# one file per 1000 images, written under a temporary name and renamed, so a stopped run leaves no half file
with ThreadPoolExecutor(12) as ex:
    for k in range(0, len(paths), 1000):
        name = "%s_%05d.npy" % (prefix, k // 1000)
        if os.path.exists(name):
            continue
        if time.time() - start > budget:
            print("stopped early, run again to resume")
            sys.exit(0)
        np.save(name + ".tmp.npy", np.stack(list(ex.map(work, paths[k:k + 1000]))))
        os.replace(name + ".tmp.npy", name)
print("all thumbnails done:", len(paths))
