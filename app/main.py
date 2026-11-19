"""Starts the desktop window: a web front end (app/frontend) in a native window, talking to app.api.Api.
Usage: python -m app.main   (from the repository root, in the conda environment)
Weights: models/v1.pt in the data folder. If it is missing, copy the file or set MRI_APP_BEST_PT to the best.pt from Kaggle."""
import os
import shutil

import webview

from .api import Api
from .backend import trainer
from .backend.settings import Settings

HERE = os.path.dirname(os.path.abspath(__file__))


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
