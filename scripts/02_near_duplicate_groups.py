"""Stage 2: near-duplicate groups.
Connects images whose dHash Hamming distance is <= THRESHOLD (union-find) and
reports group sizes and groups that mix classes.
Usage (from data/interim): python ../../scripts/02_near_duplicate_groups.py HASHES.tsv THRESHOLD OUT.tsv
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils import popcount, read_hashes

hashes, thr, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
paths, md5, dh, cls = read_hashes(hashes)
n = len(paths)

parent = list(range(n))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for i in range(n - 1):
    near = np.where(popcount(dh[i + 1:] ^ dh[i]) <= thr)[0] + i + 1
    for j in near:
        a, b = find(i), find(int(j))
        if a != b:
            parent[b] = a

root = np.array([find(i) for i in range(n)])
ids = {r: k for k, r in enumerate(sorted(set(root.tolist())))}
group = np.array([ids[r] for r in root.tolist()])
size = np.bincount(group)

with open(out, "w") as fo:
    fo.write("path\tmd5\tdhash\tclass\tgroup\tgroup_size\n")
    for i in range(n):
        fo.write("%s\t%s\t%016x\t%s\t%d\t%d\n" % (paths[i], md5[i], dh[i], cls[i], group[i], size[group[i]]))

mixed = sum(1 for g in range(len(size)) if size[g] > 1 and len({cls[i] for i in np.where(group == g)[0]}) > 1)
print("threshold", thr, "images", n, "groups", len(size), "singletons", int((size == 1).sum()),
      "largest", int(size.max()), "mixed-class groups", mixed)
