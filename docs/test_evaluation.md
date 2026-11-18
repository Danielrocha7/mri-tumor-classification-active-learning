# Test-set evaluation of the final supervised models

Date: 2026-11-18. Script: `scripts/09_test_evaluation.py`. Tables: `docs/test_summary.csv`, `docs/test_report.txt`.

## Setup

- Test set: 2,989 images from the held-out fold of the group-aware split (seed 42). Near-duplicate groups never cross between training, validation and test.
- Models were fixed before the test set was opened (see `docs/decisions.md`): ConvNeXt-Tiny (best validation) and EfficientNet-B0 with the original project's head (reference).
- Each model was trained with 3 seeds (42, 43, 44) on folds 1 to 5 of the training and validation data, with fold 0 for validation. The epoch was chosen on validation macro-F1 and the test set was predicted once per run. Nothing was changed after seeing the test results.
- Confidence intervals: 95% percentile bootstrap (2,000 resamples) that resamples near-duplicate groups instead of single images, so twin images do not make the intervals too narrow.

## Results

| Run | Best epoch | Val macro-F1 | Test accuracy (95% CI) | Test macro-F1 | Errors |
|:--|--:|--:|:--|--:|--:|
| ConvNeXt-Tiny, seed 42 | 7 | 0.9973 | 0.9987 (0.9972 to 0.9997) | 0.9987 | 4 |
| ConvNeXt-Tiny, seed 43 | 11 | 0.9967 | 0.9990 (0.9977 to 1.0000) | 0.9990 | 3 |
| ConvNeXt-Tiny, seed 44 | 5 | 0.9921 | 0.9916 (0.9857 to 0.9960) | 0.9917 | 25 |
| EfficientNet-B0 (original head), seed 42 | 11 | 0.9891 | 0.9936 (0.9899 to 0.9968) | 0.9937 | 19 |
| EfficientNet-B0 (original head), seed 43 | 10 | 0.9895 | 0.9920 (0.9881 to 0.9956) | 0.9921 | 24 |
| EfficientNet-B0 (original head), seed 44 | 7 | 0.9887 | 0.9923 (0.9886 to 0.9955) | 0.9923 | 23 |

Mean over the 3 seeds (test macro-F1): ConvNeXt-Tiny 0.9965 (SD 0.0042); EfficientNet-B0 with the original head 0.9927 (SD 0.0009).

Per class (pooled over seeds), the weakest recall is meningioma (ConvNeXt-Tiny 0.994, EfficientNet-B0 0.985); most errors for both models are meningioma predicted as pituitary (12 and 26 of the 2,205 meningioma images, pooled over seeds). Confusion matrices are in `docs/test_report.txt`.

Paired McNemar tests (exact, same seed, same test images): ConvNeXt-Tiny is better for seeds 42 (17 vs 2 discordant images, p = 7e-4) and 43 (22 vs 1, p = 6e-6) and not different for seed 44 (21 vs 23, p = 0.88).

## Reading

- Both models are very accurate on a test set with no near-duplicates of the training data. The test macro-F1 is as high as, or slightly higher than, the validation macro-F1, so there is no sign that the validation numbers were inflated by leakage.
- The advantage of ConvNeXt-Tiny over EfficientNet-B0 is not consistent across seeds. Seed 44 selected epoch 5 on validation and ended with 25 errors, as many as EfficientNet-B0. The seed-to-seed variation of ConvNeXt-Tiny (SD 0.0042) is larger than that of EfficientNet-B0. With 3 seeds, the honest summary is that ConvNeXt-Tiny is probably at least as good, and often better, but the data do not show a firm ranking.
- With 3 to 25 errors per run, the numbers are driven by a handful of images; differences in the third decimal place are not meaningful.

## Limits

- This is an internal test: the same Kaggle dataset, same acquisition style, patients unknown. Group-aware splits remove near-duplicate images but cannot guarantee that no patient appears in both training and test, because the dataset has no patient identifiers. The external test (second dataset) is still to be done and is the more informative one.
- Validation fold 0 was used to choose the epoch for all seeds, one fold only; a noisy choice can pick a poor epoch (seed 44).
- The McNemar p-values are per seed and not corrected for multiple comparisons.
- These numbers are not directly comparable with the 0.9475 reported by the original project: different test set, and the original test set had near-duplicates in training and was used to choose the checkpoint.
