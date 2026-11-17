# Curation notes (Kaggle training dataset)

Measurements behind the near-duplicate method. Numbers come from `scripts/01` to `scripts/04` run on the 22,017 images.

## Why not a hash distance

Pairs are linked only if the pixel correlation of 64x64 thumbnails is at least 0.95 (two different brain slices correlate around 0.7 by themselves). Among 46,931 linked pairs, the hash distance is spread widely:

| dHash distance | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | > 10 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Linked pairs | 9,359 | 6,451 | 6,767 | 6,014 | 6,226 | 3,954 | 3,137 | 2,011 | 1,403 | 803 | 438 | 368 |

A hash filter would miss many real duplicates, and joining by hash distance chains unrelated images (at distance 2 the largest group has 1,269 images). The hash is kept for exact duplicates and for reporting.

## Threshold sensitivity

| Correlation threshold | Groups | Singletons | Largest group | Linked pairs with different class |
|--:|--:|--:|--:|--:|
| 0.90 | 6,200 | 3,866 | 6,713 | 25,696 |
| 0.95 | 11,630 | 8,967 | 124 | 126 |
| 0.98 | 15,213 | 12,303 | 58 | 6 |

At 0.90 the groups collapse into one giant group, so it is too permissive. 0.95 is the value used. At 0.98 only 6 pairs have different labels.

## Label conflicts

At 0.98, six pairs of images with correlation 1.000 carry different labels (meningioma against pituitary). They are the same image under two classes, so they are label errors in the dataset. Of the 126 pairs at 0.95 with different labels, most still need manual review.

## Limits

- Thumbnails of 64x64 pixels and a fixed threshold: duplicates that were cropped, flipped or rotated can be missed.
- No patient identifiers exist, so groups are a proxy for patients.

## Splits (seed 42)

25 groups with more than one class (266 images) are excluded and listed in `splits/excluded_label_conflicts.tsv`. Labels are not corrected by eye: a label can only be fixed by someone qualified to read the scans. The remaining 21,751 images go to `splits/kaggle_splits_seed42.tsv`: 2,989 test and 18,762 for cross-validation (6 folds), grouped and stratified by class. `tests/test_splits.py` checks that no group crosses partitions and that no test image has a pixel twin (correlation >= 0.95) in the training part. The same check on a random per-image split finds twins for a large share of the test images, which is how the original split behaved.
