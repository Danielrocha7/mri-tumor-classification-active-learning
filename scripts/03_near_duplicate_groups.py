"""Stage 3: near-duplicate groups.
Every pair of images is compared by the pixel correlation of the 64x64
thumbnails; pairs with correlation >= MIN_CORR are linked and linked images
form groups (union-find). The dHash distance is only reported, not used.
Usage (from data/interim): python ../../scripts/03_near_duplicate_groups.py HASHES.tsv THUMBPREFIX MIN_CORR OUT.tsv
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils import load_thumbs, popcount, read_hashes

hashes, prefix, minc, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
paths, md5, dh, cls = read_hashes(hashes)
n = len(paths)
z = load_thumbs(prefix, n)
cls_arr = np.array(cls)

# union-find: every image starts as its own group
parent = list(range(n))


# root of the group of image i (with path compression)
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


# compare 1000 rows at a time against all images; the vectors are normalised, so a dot product is the Pearson correlation
by_dist = np.zeros(65, dtype=int)
cross = 0
block = 1000
for s in range(0, n, block):
    sim = z[s:s + block] @ z.T
    # pairs above the correlation threshold
    ii, jj = np.where(sim >= minc)
    ii = ii + s
    # count each pair once and drop the pair of an image with itself
    keep = jj > ii
    ii, jj = ii[keep], jj[keep]
    by_dist += np.bincount(popcount(dh[ii] ^ dh[jj]), minlength=65)
    cross += int((cls_arr[ii] != cls_arr[jj]).sum())
    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

# renumber the groups 0 to G-1
root = np.array([find(i) for i in range(n)])
ids = {r: k for k, r in enumerate(sorted(set(root.tolist())))}
group = np.array([ids[r] for r in root.tolist()])
size = np.bincount(group)

with open(out, "w") as fo:
    fo.write("path\tmd5\tdhash\tclass\tgroup\tgroup_size\n")
    for i in range(n):
        fo.write("%s\t%s\t%016x\t%s\t%d\t%d\n" % (paths[i], md5[i], dh[i], cls[i], group[i], size[group[i]]))

# groups with more than one class are label conflicts (excluded in stage 5)
mixed = sum(1 for g in range(len(size)) if size[g] > 1 and len({cls[i] for i in np.where(group == g)[0]}) > 1)
print("linked pairs", int(by_dist.sum()), "of which different class", cross)
print("linked pairs by hash distance 0..10:", by_dist[:11].tolist(), "beyond 10:", int(by_dist[11:].sum()))
print("min_corr", minc, "images", n, "groups", len(size), "singletons", int((size == 1).sum()),
      "largest", int(size.max()), "mixed-class groups", mixed)
