"""The labelled sets of every active-learning run come from the pool only (trainval, folds 1 to 5):
never from the validation fold, the test fold or the external set. Uses the tracked run outputs."""
import glob
import json
import os
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MANIFEST = os.path.join(ROOT, "splits", "kaggle_splits_seed42.tsv")
RUNS = sorted(glob.glob(os.path.join(ROOT, "results", "active_learning", "*_s*", "order.json")))

if not RUNS:
    pytest.skip("active-learning outputs missing", allow_module_level=True)

rows = [l.rstrip("\n").split("\t") for l in open(MANIFEST)][1:]
split = np.array([r[4] for r in rows])
fold = np.array([int(r[5]) for r in rows])
pool = set(np.where((split == "trainval") & (fold != 0))[0])


@pytest.mark.parametrize("path", RUNS, ids=[os.path.basename(os.path.dirname(p)) for p in RUNS])
def test_labelled_images_come_from_the_pool(path):
    order = json.load(open(path))
    assert len(order) == len(set(order)), "an image was labelled twice"
    assert set(order) <= pool, "a labelled image is outside the pool (validation, test or other)"


def test_pool_size():
    assert len(pool) == 15717
