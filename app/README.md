# App: Brain MRI Classifier

A desktop-style app: a web interface (`frontend/`, HTML/CSS/JS) in a native window (pywebview) over a Python backend.

| Part | Where |
|---|---|
| Window and start | `main.py` (`python -m app.main` from the repository root) |
| Bridge JS to Python | `api.py` |
| Accounts, history, review queue | `backend/db.py`, `backend/auth.py` (sqlite, bcrypt) |
| Inference | `backend/model_service.py` (ConvNeXt-Tiny, same preprocessing as training) |
| Retraining | `backend/retrain.py` (which images may be used), `backend/trainer.py` (CPU fine-tuning, versions) |

## Running it

1. Create the environment: `conda env create -f environment.yml`.
2. Put the trained weights in place. Either set `MRI_APP_BEST_PT` to the `best.pt` trained on Kaggle (`scripts/07_train_supervised.py --save-weights`), or copy it to `~/.mri-classifier/models/v1.pt`.
3. `python -m app.main`. Without pywebview (for example in a browser) `frontend/index.html` opens in demo mode with sample data.

User data (database, uploaded images, model versions) live in `~/.mri-classifier` (or `MRI_APP_DATA`), never in the repository.

## What the app does, and what it does not claim

- **Margin** is the difference between the two highest probabilities. The review threshold is 0.80 (`results/margin_threshold.txt`, chosen on validation probabilities only). On the external set a margin below 0.95 catches about 44 % of the errors, and most errors have a high margin, so the queue is an aid and not an error detector. Predictions can also be sent to review by hand.
- **Scores are softmax outputs**, not calibrated probabilities.
- **Retraining** fine-tunes the current model on the images reviewed since the last round (5 epochs, learning rate 1e-5, CPU, in the background). A new version is promoted only if its accuracy on images of earlier rounds does not drop by more than 5 points. Those images were training data of earlier rounds, so this checks for forgetting, not generalisation. The reviewed images are also the hard ones, which biases the training set.
- **Test and external images never train the model**: an image whose md5 is in the test split of `splits/kaggle_splits_seed42.tsv` is dropped before training. The external set has no tracked md5, so it cannot be blocked this way, and near-duplicates are not caught.
- Research prototype, not a clinical tool.

## Tests

`pytest tests/test_app_backend.py tests/test_api.py` need no torch. `tests/test_model_service.py` and `tests/test_trainer.py` need a working torch and skip otherwise.
