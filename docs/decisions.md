# Decision log

| Date | Decision | Reason | Evidence | Status |
|:--|:--|:--|:--|:--|
| 2026-11-17 | New public repository with a fresh history, not a fork of the private group repository | The old repository contains student numbers, a user database and data files | Pre-publication check | Planned |
| 2026-11-17 | Original group code kept in `original/`; new work lives outside it | Makes clear what is group work and what is new | `README.md` | Planned |
| 2026-11-17 | Images, model weights, databases and run-time files are not tracked | The dataset license is "Unknown" on Kaggle; the database holds user accounts | `data/README.md`, `.gitignore` | Adopted |
| 2026-11-17 | README and documentation in English | Portfolio audience | - | Adopted |
| 2026-11-17 | Scripts: short header, flat structure, short comments; results confirmed identical after any rewrite of an original script | Easier to read and to compare with the originals | - | Adopted |
| 2026-11-17 | Code license: MIT | Permissive and common for portfolio code | Needs the written agreement of the other two original authors | Pending |
| 2026-11-17 | Train on the same Kaggle dataset as the original work; internal blind test on new group-aware splits; external test on a second dataset never used for training | Keeps continuity with the original and separates in-dataset from out-of-dataset performance | `pipeline.md` | Adopted |
| 2026-11-17 | Report three protocols side by side: replica of the original, same model on curated splits, external test | The gap between the first two measures how much of the original result came from data leakage; this is a project objective | `evaluation_audit.md` | Adopted |
| 2026-11-17 | External test candidates: Nickparvar (Kaggle) and BRISC, used locally only and never redistributed | Both are tested for hash overlap with the training data first; overlapping images are removed from the external set | Overlap measured in phase 1 | Planned |
| 2026-11-17 | External test set = only Nickparvar images with no near-twin in the training data (pixel correlation < 0.95) | 71 % of the Nickparvar images have a near-twin in the Kaggle dataset, so the full dataset is not external | `data/README.md` | Adopted |
| 2026-11-17 | Near-duplicate groups from exhaustive pixel correlation of 64x64 thumbnails, threshold 0.95; dHash not used as a filter | A hash-distance window misses real duplicates (confirmed pairs still appear at distances 5 to 10), and larger distances chain unrelated images | `docs/curation.md` | Adopted |
| 2026-11-17 | Splits: groups with mixed labels excluded; one fold of 7 as test (seed 42), the rest in 6 cross-validation folds; groups never cross partitions | Avoids leakage and label conflicts; labels are not changed by eye | `splits/`, `tests/test_splits.py` | Adopted |

| 2026-11-18 | Skip batches with fewer than two images in supervised training | BatchNorm1d in the original head fails on a batch of one (fold 3 leaves one) | commit 51d2c39 | Adopted |
| 2026-11-18 | ConvNeXt-Tiny is the reference supervised baseline for the next phases | Best mean validation macro-F1 over 6 folds (0.9970), small margin | `docs/baselines.md` | Adopted |
| 2026-11-18 | Checkpoints chosen on validation macro-F1; test set used once per final model | Avoids repeating the test-set selection of the original | `docs/baselines.md` | Adopted |
| 2026-11-18 | DINOv2-B linear probe reported in the table but not in the figure | Its macro-F1 (0.919) is far below the fine-tuned models and would distort the axes | `docs/baselines.md` | Adopted |
| 2026-11-18 | Test set opened once per final model, with the models chosen before seeing any test result: ConvNeXt-Tiny (best validation) and EfficientNet-B0 with the original head (reference for the comparison with the original project) | Avoids choosing models on the test set, which the original did | `docs/baselines.md` | Adopted |
| 2026-11-18 | Test evaluation: 3 seeds (42, 43, 44) per model, validation fold 0 for checkpoint selection, `--predict-test`; no other model or setting is evaluated on test | Seed variation is reported; the test result cannot influence any later choice | `scripts/07_train_supervised.py` | Adopted |
| 2026-11-18 | Test set opened for the 6 planned runs; results reported as obtained, with group bootstrap CIs and McNemar per seed (`scripts/09_test_evaluation.py`) | Follows the design fixed before the test was opened | `docs/test_evaluation.md` | Adopted |
| 2026-11-18 | External test: the 6 runs (ConvNeXt-Tiny and EfficientNet-B0 original head, seeds 42-44) are retrained with the same recipe and predict the 2,180 external images at the best-validation epoch (`--external`); weights saved (`--save-weights`) | The training script kept no checkpoints, so the external set cannot be scored on the earlier runs | `scripts/07_train_supervised.py` | Adopted |
| 2026-11-18 | Caveat for the external set: it only keeps Nickparvar images with no near-twin in the Kaggle data (71 % of the dataset overlaps), so it is a harder, selected subset and only partly independent | The overlap shows both datasets share sources | `data/README.md` | Adopted |
