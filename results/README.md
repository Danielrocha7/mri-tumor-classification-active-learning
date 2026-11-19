# Results of the final runs

`final_runs/<model>_s<seed>/` holds, for each of the six runs (ConvNeXt-Tiny and EfficientNet-B0 with the original head, seeds 42 to 44): `metrics.json` (arguments, per-epoch history, best epoch, metrics), and the class probabilities for the validation fold, the internal test fold and the external set (`val_probs.npz`, `test_probs.npz`, `external_probs.npz`, with the row indices into the packed data).

These are predictions only. No images and no weights are stored here. The labels come from the packed arrays that `scripts/06_pack_images.py` builds from the manifests in `splits/`, so `scripts/09_test_evaluation.py` and `scripts/10_external_evaluation.py` can recompute the tables in `docs/` once the datasets are downloaded and packed.

The runs were produced on Kaggle (T4) with `scripts/07_train_supervised.py --predict-test --external --save-weights`.

# Results of the active-learning runs

`active_learning/<arm>_s<seed>/` holds one folder per run (arms: random, entropy, margin, mcdropout, coreset, badge, original, full; seeds 42 to 44): `rounds.json` (per-round validation metrics, selection and training time, class counts of the labelled set), `order.json` (labelled indices in the order they were chosen), `done.json`, and test and external class probabilities at the saved budgets (`test_probs_<n>.npz`, `external_probs_<n>.npz`; 60, 300, 600 and 1,200 labelled images, and the last round for the original-protocol arm). `active_learning/labels.npz` has the test and external labels and the class names, so `scripts/14_active_learning_analysis.py` runs without the images.

Predictions and logs only; no images and no weights. Produced on Kaggle (T4) with `scripts/13_active_learning.py`.
