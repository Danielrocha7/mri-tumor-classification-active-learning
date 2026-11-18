"""Stage 1: inventory and hashing of an image folder.
Writes one row per image: relative path, MD5, 64-bit dHash, width, height.
Resumable. Usage (from data/interim): python ../../scripts/01_hash_images.py ROOT OUT.tsv
"""
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils import dhash

root, out = sys.argv[1], sys.argv[2]
exts = (".png", ".jpg", ".jpeg")

# resume support: skip files that are already in the output table
done = set()
if os.path.exists(out):
    done = {l.split("\t")[0] for l in open(out)}
else:
    open(out, "w").write("path\tmd5\tdhash\twidth\theight\n")

# image files that still need hashing
files = []
for r, _, fs in os.walk(root):
    for f in fs:
        if f.lower().endswith(exts):
            rel = os.path.relpath(os.path.join(r, f), root).replace(os.sep, "/")
            if rel not in done:
                files.append(rel)
files.sort()


# the MD5 finds exact copies; the dHash (src/utils.py) finds visually similar images
def work(rel):
    data = open(os.path.join(root, rel), "rb").read()
    value, w, h = dhash(data)
    return "%s\t%s\t%016x\t%d\t%d\n" % (rel, hashlib.md5(data).hexdigest(), value, w, h)


# hash in parallel; rows are appended as they finish, so a stopped run can be resumed
with open(out, "a") as fo, ThreadPoolExecutor(12) as ex:
    for i, line in enumerate(ex.map(work, files)):
        fo.write(line)
        if i % 500 == 0:
            fo.flush()
print("hashed", len(files), "new files")
