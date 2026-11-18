"""Shared helpers: difference hash, thumbnails, Hamming distance, seed."""
import io
import random
import numpy as np
from PIL import Image


def dhash(data):
    """64-bit difference hash (9x8 grayscale) of an image given as bytes."""
    im = Image.open(io.BytesIO(data)).convert("L")
    w, h = im.size
    px = np.asarray(im.resize((9, 8), Image.LANCZOS)).astype(int)
    # a bit is 1 when a pixel is brighter than its right neighbour
    bits = (px[:, :-1] > px[:, 1:]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value, w, h


def thumb(data):
    """64x64 grayscale thumbnail (uint8) of an image given as bytes."""
    im = Image.open(io.BytesIO(data)).convert("L").resize((64, 64), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def load_thumbs(prefix, n):
    """Load the thumbnail chunks written by 02_make_thumbnails.py as normalised vectors."""
    parts = []
    for k in range(0, n, 1000):
        parts.append(np.load("%s_%05d.npy" % (prefix, k // 1000)))
    x = np.concatenate(parts).reshape(n, -1).astype(np.float32)
    # centre and scale every image, so that a dot product between two images is their Pearson correlation
    x -= x.mean(axis=1, keepdims=True)
    norm = np.sqrt((x ** 2).sum(axis=1, keepdims=True))
    # a flat image gets a zero vector instead of a division by zero
    norm[norm == 0] = np.inf
    return x / norm


def popcount(x):
    """Bit count of a uint64 array."""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0f0f0f0f0f0f0f0f)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int16)


def read_hashes(path):
    """Read a hash table: returns paths, md5, dhash (uint64), classes."""
    rows = [l.rstrip("\n").split("\t") for l in open(path)][1:]
    paths = [r[0] for r in rows]
    md5 = [r[1] for r in rows]
    dh = np.array([int(r[2], 16) for r in rows], dtype=np.uint64)
    cls = [r[0].split("/")[-2] for r in rows]
    return paths, md5, dh, cls


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
