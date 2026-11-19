"""Downloads the published model weights (a GitHub release asset) into the app's data folder.

Usage: python -m app.weights
The file is checked against a pinned SHA-256 before it is installed, so a damaged or swapped file is refused.
If you replace the published file, update WEIGHTS_SHA256 (sha256sum best.pt). MRI_APP_WEIGHTS_URL overrides the address."""
import hashlib
import os
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WEIGHTS_URL = "https://github.com/Danielrocha7/mri-tumor-classification-active-learning/releases/download/v1.0/best.pt"
WEIGHTS_SHA256 = "33e280ce1dcba71a9111743b1905ab7b8838dcbf436912edfa8090c553d7c39d"


def download_weights(settings, url=None, sha256=None):
    """Installs models/v1.pt in the data folder and returns its path. Keeps an existing v1.pt untouched."""
    url = url or os.environ.get("MRI_APP_WEIGHTS_URL") or WEIGHTS_URL
    sha256 = sha256 or WEIGHTS_SHA256
    settings.ensure_dirs()
    dest = os.path.join(settings.weights_dir, "v1.pt")
    if os.path.exists(dest):
        return dest
    fd, tmp = tempfile.mkstemp(dir=settings.weights_dir, suffix=".part")
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as r:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError("checksum mismatch: the downloaded file is not the published weights")
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return dest


def main():
    from app.backend.settings import Settings
    s = Settings()
    print("Downloading the weights to", s.weights_dir, "...")
    print("Installed:", download_weights(s))


if __name__ == "__main__":
    main()
