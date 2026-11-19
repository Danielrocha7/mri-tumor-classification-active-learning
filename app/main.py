"""Starts the desktop window: a web front end (app/frontend) in a native window, talking to app.api.Api.
Usage: open this file in VS Code and press Run, or `python app/main.py` / `python -m app.main` from the repository root.
Weights: models/v1.pt in the data folder. If it is missing, copy the file or set MRI_APP_BEST_PT to the best.pt from Kaggle."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repository root, so that `app` is importable when this file is run directly



def load_env_file(path):
    """Reads KEY=VALUE lines of a .env file into os.environ (existing variables win). No dependency needed."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env_file(os.path.join(os.path.dirname(HERE), ".env"))

import webview  # noqa: E402

from app.api import Api  # noqa: E402
from app.backend import trainer  # noqa: E402
from app.backend.settings import Settings  # noqa: E402


def main():
    settings = Settings()
    settings.blocked_manifest = os.environ.get("MRI_APP_MANIFEST") or os.path.join(HERE, "..", "splits", "kaggle_splits_seed42.tsv")
    settings.ensure_dirs()
    best = os.environ.get("MRI_APP_BEST_PT")
    if best and os.path.exists(best) and trainer.latest(settings)[1] is None:
        trainer.install_initial(settings, best)
    api = Api(settings)
    webview.create_window("Brain MRI Classifier", os.path.join(HERE, "frontend", "index.html"), js_api=api,
                          width=1280, height=800, min_size=(1000, 640))
    webview.start(storage_path=os.path.join(settings.data_dir, "webview"))  # keeps theme and language between runs


if __name__ == "__main__":
    main()
