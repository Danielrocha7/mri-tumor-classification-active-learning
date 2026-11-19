"""Starts the desktop window: a web front end (app/frontend) in a native window, talking to app.api.Api.
Usage: open this file in VS Code and press Run, or `python app/main.py` / `python -m app.main` from the repository root.
If the native window does not work on your machine (no GUI toolkit, or Windows blocks its DLLs), use app/serve.py,
which shows the same app in the browser.
Weights: models/v1.pt in the data folder. If it is missing, copy the file or set MRI_APP_BEST_PT (in .env) to the best.pt from Kaggle."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repository root, so that `app` is importable when this file is run directly

from app.envfile import load_env_file  # noqa: E402

load_env_file(os.path.join(os.path.dirname(HERE), ".env"))


def main():
    import webview
    from app.api import Api
    from app.backend import trainer
    from app.backend.settings import Settings

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
