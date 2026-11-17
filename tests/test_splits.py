"""Leakage tests for the Kaggle splits. Need the local files in data/interim (not tracked)."""
import os
import sys
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from utils import load_thumbs

MANIFEST = os.path.join(ROOT, "splits", "kaggle_splits_seed42.tsv")
GROUPS = os.path.join(ROOT, "data", "interim", "groups_kaggle_c095.tsv")
THUMBS = os.path.join(ROOT, "data", "interim", "thumbs_kaggle")

if not os.path.exists(GROUPS):
    pytest.skip("local curation files missing", allow_module_level=True)


def table(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path)]
    head = rows[0]
    return [dict(zip(head, r)) for r in rows[1:]]


@pytest.fixture(scope="module")
def manifest():
    return table(MANIFEST)


@pytest.fixture(scope="module")
def groups():
    return table(GROUPS)


def conflict_groups(groups):
    by = {}
    for r in groups:
        by.setdefault(r["group"], set()).add(r["class"])
    return {g for g, c in by.items() if len(c) > 1}


def test_every_kept_image_appears_exactly_once(manifest, groups):
    bad = conflict_groups(groups)
    expected = sorted(r["path"] for r in groups if r["group"] not in bad)
    assert sorted(r["path"] for r in manifest) == expected


def test_label_conflict_groups_are_excluded(manifest, groups):
    bad = conflict_groups(groups)
    assert len(bad) > 0
    assert not {r["group"] for r in manifest} & bad


def test_no_group_is_in_both_test_and_trainval(manifest):
    test = {r["group"] for r in manifest if r["split"] == "test"}
    trainval = {r["group"] for r in manifest if r["split"] == "trainval"}
    assert test and trainval
    assert not test & trainval


def test_no_group_is_in_two_folds(manifest):
    folds = {}
    for r in manifest:
        if r["split"] == "trainval":
            folds.setdefault(r["group"], set()).add(r["fold"])
    assert all(len(f) == 1 for f in folds.values())


def test_every_class_is_in_test_and_in_every_fold(manifest):
    classes = {r["class"] for r in manifest}
    assert len(classes) == 4
    assert {r["class"] for r in manifest if r["split"] == "test"} == classes
    for f in range(6):
        assert {r["class"] for r in manifest if r["split"] == "trainval" and r["fold"] == str(f)} == classes


def test_test_share_is_about_one_seventh(manifest):
    share = sum(r["split"] == "test" for r in manifest) / len(manifest)
    assert 0.12 <= share <= 0.17


def test_no_test_image_is_a_pixel_twin_of_a_trainval_image(manifest, groups):
    if not os.path.exists(THUMBS + "_00000.npy"):
        pytest.skip("thumbnails missing")
    idx = {r["path"]: k for k, r in enumerate(groups)}
    z = load_thumbs(THUMBS, len(groups))
    test = [idx[r["path"]] for r in manifest if r["split"] == "test"]
    trainval = [idx[r["path"]] for r in manifest if r["split"] == "trainval"]
    best = (z[test] @ z[trainval].T).max(axis=1)
    assert float(best.max()) < 0.95
