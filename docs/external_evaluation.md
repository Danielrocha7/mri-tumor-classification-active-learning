# External-set evaluation

Date: 2026-11-18. Script: `scripts/10_external_evaluation.py`. Tables: `docs/external_summary.csv`, `docs/external_report.txt`.

## Setup

- External set: 2,180 images of the Kaggle dataset "Brain Tumor MRI Dataset" (Nickparvar) that have no near-twin in the training data (pixel correlation below 0.95 against every training image). Classes: glioma 454, meningioma 531, no tumour 991, pituitary 204. The images were packed with the same preprocessing as the training data (grayscale, 224x224, bilinear).
- Models: the same six runs as in the internal test (ConvNeXt-Tiny and EfficientNet-B0 with the original head, seeds 42 to 44). The training script kept no weights, so the six runs were trained again with the same recipe and seeds and predicted the external set at the best-validation epoch. The internal test results of the repeat are identical to the first runs (same accuracy and macro-F1 in all six), so the two evaluations refer to the same models.
- Nothing was tuned on the external set: no model, threshold or preprocessing choice depends on it.
- Confidence intervals: 95% percentile bootstrap over images (2,000 resamples). There are no near-duplicate groups to resample here, because the twins were removed.

## Results

| Model | Internal test macro-F1 | External macro-F1 (3 seeds, SD) | External accuracy | Accuracy reweighted to the internal class mix |
|:--|--:|:--|--:|--:|
| ConvNeXt-Tiny | 0.9965 | 0.9495 (0.0080) | 0.9575 | 0.9452 |
| EfficientNet-B0 (original head) | 0.9927 | 0.9257 (0.0090) | 0.9378 | 0.9385 |

Per run (external macro-F1 with 95% CI): ConvNeXt-Tiny 0.9551 (0.9452 to 0.9644), 0.9532 (0.9420 to 0.9634), 0.9403 (0.9294 to 0.9514); EfficientNet-B0 0.9356 (0.9237 to 0.9462), 0.9231 (0.9098 to 0.9350), 0.9182 (0.9051 to 0.9305).

Recall per class (mean over the 3 seeds):

| Class | ConvNeXt-Tiny | EfficientNet-B0 |
|:--|--:|--:|
| glioma | 0.859 | 0.854 |
| meningioma | 0.994 | 0.986 |
| no tumour | 0.985 | 0.944 |
| pituitary (204 images) | 0.948 | 0.969 |

Paired McNemar tests (exact, same seed): ConvNeXt-Tiny is better in all three seeds (62 vs 22, 67 vs 16, 60 vs 22 discordant images; p between 1e-5 and 1e-8, not corrected for the three comparisons).

## Reading

- Performance drops on images from another source, as expected: macro-F1 falls by about 5 points for ConvNeXt-Tiny and about 7 points for EfficientNet-B0. The internal test (0.99+) therefore measures in-dataset performance and overstates what to expect on other sources.
- Most of the drop is one class. About 14% of the glioma images are misclassified by both architectures (mostly as meningioma, then as no tumour); the other classes keep recall of 0.94 or more, except no tumour for EfficientNet-B0 (0.944; mostly predicted as pituitary). Because both architectures fail in the same way on glioma, a shift between the two datasets is more likely than a weakness of one model. The cause was not investigated: acquisition differences, label conventions or the selection of "no near-twin" images are all candidates, and none is tested.
- The ranking of the two models is the same as in the internal test, and here it is consistent across seeds. ConvNeXt-Tiny generalises better on this set.
- The accuracy of the external set is not directly comparable with the internal one because the class mix differs (no tumour 45% against 23%, pituitary 9% against 28%). Macro-F1, recall per class and the reweighted accuracy are the fairer comparisons; the reweighted accuracy applies the per-class recall to the class mix of the internal test.

## Limits

- The external set is a selected, harder subset: only the Nickparvar images with no near-twin in the Kaggle data (30% of the dataset). 71% of the dataset overlaps with the training data, so the two datasets share sources and the set is only partly independent.
- Pituitary has 204 images, so its recall has a wide uncertainty (a few percentage points).
- The six runs are three seeds per model; the intervals describe test-set sampling, not training variability beyond those seeds.
- This is still not a clinical validation: no patient identifiers, no independent acquisition site.
