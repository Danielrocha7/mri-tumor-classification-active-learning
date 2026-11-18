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
- Most of the drop is one class. About 14% of the glioma images are misclassified by both architectures (mostly as meningioma, then as no tumour); the other classes keep recall of 0.94 or more, except no tumour for EfficientNet-B0 (0.944; mostly predicted as pituitary). Both architectures fail in the same way on glioma, which points at the data rather than at one model, but this is not tested. See the section on the glioma errors below.
- The ranking of the two models is the same as in the internal test, and here it is consistent across seeds. ConvNeXt-Tiny generalises better on this set.
- The accuracy of the external set is not directly comparable with the internal one because the class mix differs (no tumour 45% against 23%, pituitary 9% against 28%). Macro-F1, recall per class and the reweighted accuracy are the fairer comparisons; the reweighted accuracy applies the per-class recall to the class mix of the internal test.

## Recall per class, internal against external

Mean of the three seeds. The internal interval resamples near-duplicate groups, the external one resamples images; 2,000 resamples, the same resample shared by the three seeds of a model. Figure: `docs/figures/internal_vs_external.png`; table: `docs/internal_vs_external_recall.csv`.

| Class (n internal / n external) | ConvNeXt-Tiny internal | ConvNeXt-Tiny external (95% CI) | EfficientNet-B0 internal | EfficientNet-B0 external (95% CI) |
|:--|--:|:--|--:|:--|
| glioma (754 / 454) | 0.997 | 0.859 (0.829 to 0.891) | 0.996 | 0.854 (0.824 to 0.884) |
| meningioma (735 / 531) | 0.994 | 0.994 (0.987 to 0.999) | 0.986 | 0.986 (0.978 to 0.994) |
| no tumour (678 / 991) | 0.995 | 0.985 (0.979 to 0.991) | 0.992 | 0.944 (0.929 to 0.957) |
| pituitary (822 / 204) | 0.999 | 0.948 (0.916 to 0.975) | 0.997 | 0.969 (0.948 to 0.986) |

The glioma recall falls by about 14 points for both models. Meningioma does not change. Pituitary falls by 5 points for ConvNeXt-Tiny and 3 for EfficientNet-B0, and its interval is about 6 points wide because the external set has only 204 pituitary images; the two models are not separated on that class.

## The external glioma errors

`scripts/12_error_examples.py`. Of the 454 external gliomas, 46 are missed by all six runs, 371 are found by all six and 37 are found by some runs only. Figure: `docs/figures/glioma_examples.png` (eight random images from each of the first two groups). Image statistics (grayscale 0 to 255, foreground = pixels above 20), table `docs/figures/glioma_image_stats.csv`:

| Group | n | Mean intensity | Foreground fraction |
|:--|--:|--:|--:|
| External, missed by all six runs | 46 | 70.5 | 0.68 |
| External, found by all six runs | 371 | 36.0 | 0.48 |
| Internal test gliomas | 754 | 26.0 | 0.37 |

What can be said: the missed gliomas are brighter and fill more of the frame than the ones that are found, and the external gliomas as a whole are brighter and fill more of the frame than the internal ones. In the eight examples shown, the missed images are mostly axial or sagittal, tightly cropped, with a bright, often ring-shaped lesion (one is T2-weighted), while the found ones are more often coronal or wide-field with the neck visible. The missed ones are predicted as meningioma in most cases and as no tumour in some.

What cannot be said: that framing, brightness or slice plane causes the errors. The statistics are descriptive, the examples are eight images per group, and only 46 images are missed by every run. A cause would need a controlled test (for example normalising intensity and cropping, or stratifying by plane) and, since this is the external set, nothing found here is used to change the models.

## Limits

- The external set is a selected, harder subset: only the Nickparvar images with no near-twin in the Kaggle data (30% of the dataset). 71% of the dataset overlaps with the training data, so the two datasets share sources and the set is only partly independent.
- Pituitary has 204 images, so its recall has a wide uncertainty (a few percentage points).
- The six runs are three seeds per model; the intervals describe test-set sampling, not training variability beyond those seeds.
- This is still not a clinical validation: no patient identifiers, no independent acquisition site.
