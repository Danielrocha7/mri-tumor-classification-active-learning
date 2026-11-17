"""Stage 6: pack the split images into one array for fast training on Colab.
Grayscale, resized to 224x224 (bilinear, as the original), uint8, in the order of
the split manifest. Written in chunks; stops after SECONDS and can be run again.
Usage (from data/interim): python ../../scripts/06_pack_images.py ROOT MANIFEST.tsv OUT.npz SECONDS
"""
import os
import sys
import time
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

root, manifest, out, budget = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
rows = [l.rstrip("\n").split("\t") for l in open(manifest)][1:]
chunk = 2000
prefix = out.replace(".npz", "_chunk")
start = time.time()


def load(r):
    im = Image.open(os.path.join(root, r[0])).convert("L").resize((224, 224), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


with ThreadPoolExecutor(12) as ex:
    for k in range(0, len(rows), chunk):
        name = "%s_%03d.npy" % (prefix, k // chunk)
        if os.path.exists(name):
            continue
        if time.time() - start > budget:
            print("stopped early, run again to resume")
            sys.exit(0)
        np.save(name + ".tmp.npy", np.stack(list(ex.map(load, rows[k:k + chunk]))))
        os.replace(name + ".tmp.npy", name)

images = np.concatenate([np.load("%s_%03d.npy" % (prefix, k // chunk)) for k in range(0, len(rows), chunk)])
classes = sorted({r[2] for r in rows})
np.savez(out, images=images,
                    y=np.array([classes.index(r[2]) for r in rows], dtype=np.int64),
                    split=np.array([r[4] for r in rows]),
                    fold=np.array([int(r[5]) for r in rows]),
                    group=np.array([r[3] for r in rows]),
                    paths=np.array([r[0] for r in rows]),
                    classes=np.array(classes))
print("packed", images.shape, "classes", classes)
