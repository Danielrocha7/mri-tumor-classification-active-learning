"""Stage 17: check a local image folder against a manifest of the splits.
Compares every manifest row with the file on disk (present? same MD5?) and counts files the manifest does not list.
Exit code 1 if any file is missing or different. Rows without a hash ("-") only check that the file exists.
Usage: python scripts/17_check_data.py IMAGE_ROOT MANIFEST.tsv [MORE_MANIFESTS.tsv ...]
Example: python scripts/17_check_data.py data/raw/kaggle_brain_tumour_classification splits/kaggle_splits_seed42.tsv splits/excluded_label_conflicts.tsv
"""
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor

root, manifests = sys.argv[1], sys.argv[2:]
exts = (".png", ".jpg", ".jpeg")

# first manifest: the rows to verify; the others only mark files as known (for example the excluded ones)
expected, known = {}, set()
for k, m in enumerate(manifests):
    rows = [l.rstrip("\n").split("\t") for l in open(m)]
    head = rows[0]
    for r in rows[1:]:
        row = dict(zip(head, r))
        known.add(row["path"])
        if k == 0:
            expected[row["path"]] = row["md5"]


# MD5 of the file, or None when it does not exist
def check(rel):
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        return None
    return hashlib.md5(open(p, "rb").read()).hexdigest()


paths = sorted(expected)
with ThreadPoolExecutor(12) as ex:
    found = dict(zip(paths, ex.map(check, paths)))

missing = [p for p in paths if found[p] is None]
different = [p for p in paths if found[p] is not None and expected[p] != "-" and found[p] != expected[p]]
no_hash = [p for p in paths if found[p] is not None and expected[p] == "-"]
ok = len(paths) - len(missing) - len(different)

# image files on disk that no manifest lists
extra = 0
for r, _, fs in os.walk(root):
    for f in fs:
        if f.lower().endswith(exts) and os.path.relpath(os.path.join(r, f), root).replace(os.sep, "/") not in known:
            extra += 1

print("matching the manifest: %d of %d" % (ok, len(paths)))
print("missing: %d" % len(missing))
print("different: %d" % len(different))
print("no hash to compare: %d" % len(no_hash))
print("not in the manifest: %d" % extra)
for p in (missing + different)[:5]:
    print("  e.g. " + p)
sys.exit(1 if missing or different else 0)
