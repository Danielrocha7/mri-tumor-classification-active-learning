"""Tests for scripts/17_check_data.py on a tiny fake image folder (no real data needed)."""
import hashlib
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SCRIPT = os.path.join(ROOT, "scripts", "17_check_data.py")


def md5(b):
    return hashlib.md5(b).hexdigest()


def make(tmp_path, files, manifest_rows):
    """Write files {relative path: bytes} and a manifest of (path, md5) rows."""
    img = tmp_path / "img"
    for rel, data in files.items():
        p = img / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    man = tmp_path / "manifest.tsv"
    man.write_text("path\tmd5\tclass\n" + "".join("%s\t%s\tx\n" % r for r in manifest_rows))
    return str(img), str(man)


def run(img, *manifests):
    return subprocess.run([sys.executable, SCRIPT, img, *manifests], capture_output=True, text=True)


def test_identical_set_passes(tmp_path):
    files = {"a/1.png": b"one", "a/2.png": b"two"}
    img, man = make(tmp_path, files, [(k, md5(v)) for k, v in files.items()])
    r = run(img, man)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2 of 2" in r.stdout


def test_missing_file_fails(tmp_path):
    img, man = make(tmp_path, {"a/1.png": b"one"}, [("a/1.png", md5(b"one")), ("a/2.png", md5(b"two"))])
    r = run(img, man)
    assert r.returncode == 1
    assert "missing: 1" in r.stdout


def test_changed_content_fails(tmp_path):
    img, man = make(tmp_path, {"a/1.png": b"CHANGED"}, [("a/1.png", md5(b"one"))])
    r = run(img, man)
    assert r.returncode == 1
    assert "different: 1" in r.stdout


def test_extra_files_are_reported_but_do_not_fail(tmp_path):
    img, man = make(tmp_path, {"a/1.png": b"one", "a/9.png": b"extra"}, [("a/1.png", md5(b"one"))])
    r = run(img, man)
    assert r.returncode == 0
    assert "not in the manifest: 1" in r.stdout


def test_second_manifest_counts_as_known(tmp_path):
    img, man = make(tmp_path, {"a/1.png": b"one", "a/9.png": b"excl"}, [("a/1.png", md5(b"one"))])
    other = tmp_path / "other.tsv"
    other.write_text("path\tmd5\tclass\na/9.png\t%s\tx\n" % md5(b"excl"))
    r = run(img, man, str(other))
    assert r.returncode == 0
    assert "not in the manifest: 0" in r.stdout


def test_manifest_without_md5_only_checks_presence(tmp_path):
    img, man = make(tmp_path, {"a/1.png": b"anything"}, [("a/1.png", "-")])
    r = run(img, man)
    assert r.returncode == 0
    assert "no hash to compare: 1" in r.stdout
