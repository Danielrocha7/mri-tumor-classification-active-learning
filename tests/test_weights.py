"""Downloading the published weights: checksum is verified, a bad file is never installed, an existing file is kept."""
import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from app.backend.settings import Settings
from app.weights import download_weights


@pytest.fixture
def setup(tmp_path):
    payload = b"fake weights " * 1000
    src = tmp_path / "best.pt"
    src.write_bytes(payload)
    s = Settings(data_dir=str(tmp_path / "data"))
    return s, src.as_uri(), hashlib.sha256(payload).hexdigest(), payload


def test_download_installs_v1_when_the_checksum_matches(setup):
    s, url, digest, payload = setup
    dest = download_weights(s, url=url, sha256=digest)
    assert dest == os.path.join(s.weights_dir, "v1.pt")
    assert open(dest, "rb").read() == payload
    assert not [f for f in os.listdir(s.weights_dir) if f.endswith(".part")]


def test_wrong_checksum_installs_nothing_and_leaves_no_partial_file(setup):
    s, url, _, _ = setup
    with pytest.raises(ValueError, match="checksum"):
        download_weights(s, url=url, sha256="0" * 64)
    assert os.listdir(s.weights_dir) == []


def test_existing_weights_are_not_downloaded_again(setup):
    s, _, digest, _ = setup
    s.ensure_dirs()
    existing = os.path.join(s.weights_dir, "v1.pt")
    open(existing, "wb").write(b"mine")
    assert download_weights(s, url="file:///does/not/exist", sha256=digest) == existing
    assert open(existing, "rb").read() == b"mine"
