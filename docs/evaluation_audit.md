# Audit of the original evaluation

Date: 2026-11-17. Scope: the code, split files and saved state of the original group project, read as they are. No model was retrained or re-evaluated, and the saved weights were not run.

## Method

- All 22,017 images in the original `Dataset Treino` folder were hashed: MD5 of the file bytes, and a 64-bit difference hash (grayscale, resized to 9x8, one bit per horizontal gradient).
- The partitions were rebuilt from `test_files.npy`, `treino_rotulado.csv`, `treino_nao_rotulado.csv` and the saved indices (`state_initial_indices.npy`, `state_pool_indices.npy`).
- The hash was checked against pixels: for a sample of test images and their closest training image, the correlation of the 64x64 grayscale versions was computed and compared with random pairs of the same class.

## Findings

| # | Finding | Evidence |
|:--|:--|:--|
| 1 | The test set is not independent of the training data: no byte-identical copies, but many near-duplicates | 25.7% of test images have a training image of the same class with an identical hash (a different class: 0.1%); 49.5% within Hamming distance 2; 68.3% within 4. In a sample of 60 pairs at distance 0, the median correlation was 1.000 and 90% were above 0.98; random same-class pairs had a median of 0.476 |
| 2 | The test set was used to choose the model | `TrainCNN3.py` and `retrain.py` keep the checkpoint with the best test accuracy, and the early stopping counts non-improvements on the test set. Best 0.9475 against 0.9255 for the last of 31 recorded evaluations |
| 3 | Saved indices are read against the wrong list | The indices refer to the training split (17,613 images). `config.py` and `retrain.py` apply them to the full folder (22,017 images). As a result 129 of the 596 listed labelled images are test images, and the three CSV files add up to 22,146 = 22,017 + 129. `retrain.py` builds its base training set from these indices, so a retraining step trains on test images and then reports test accuracy |
| 4 | An application upload can come from the test set | Of the 2 images in `Code/pendentes`, 1 is a test image. The 88 images in `Code/processadas` were not in the test set (87 in the pool, 1 in the labelled set) |
| 5 | Active learning skews the labelled set | Final labelled set: 38 pituitary images out of 596 (6%), against 26% of the dataset |
| 6 | Only accuracy is stored | The per-class report and the confusion matrix are printed but not saved; the saved history holds accuracy only. There is no validation set, no random-sampling baseline and no fully supervised baseline |
| 7 | The dataset does not match its Kaggle description | 22,017 PNG images in the folder against 18,279 JPG images on the page; the page lists the license as "Unknown" |

## Limits of this audit

- The difference hash is not invariant to flips, rotations or rescaling, so finding 1 is a lower bound.
- The pixel check used 60 pairs per distance group.
- There are no patient identifiers, so leakage between slices of the same patient cannot be ruled out or measured.
- Finding 3 concerns the retraining script and the CSV files. The first training script (`TrainCNN3.py`) uses the indices consistently with the training split, so it does not train on test images directly.
- The saved weights were not evaluated, so how much of the reported accuracy comes from findings 1 and 2 is not measured here.

## What follows from this

The reported 0.9475 is not a reliable estimate of performance on new images. The re-evaluation will group near-duplicate images before splitting, keep the test set untouched until the end, choose checkpoints on a separate validation set, and report per-class metrics.
