# Data

Data files are **not** stored in Git (the `data/` folder is in `.gitignore`). This file records where they come from, what was checked and where they are kept locally.

## Sources

| Source | Version / date checked | License | Notes |
|:--|:--|:--|:--|
| Kaggle, [Brain Tumour Classification](https://www.kaggle.com/datasets/rishiksaisanthosh/brain-tumour-classification) (uploader: Rishik Sai Santhosh) | Page checked on 2026-11-17; shown as updated about 2 years earlier | "Unknown" on the Kaggle page | 4 classes: glioma, meningioma, pituitary, no tumor. The page states 18,279 images in JPG format and gives no origin for them. |

## To reconcile

- The folder used by the original project holds 22,017 PNG images (5,284 glioma, 5,332 meningioma, 5,621 no tumor, 5,780 pituitary), not 18,279 JPG images, so it is not known to match the current Kaggle version.
- The page does not say where the images come from. Public brain MRI datasets on Kaggle are often combinations of earlier ones, which can put the same image on both sides of a split. A first check found no byte-identical files but many near-duplicates between the original training and test sets (see `docs/evaluation_audit.md`), so the near-duplicate groups must be kept together in any new split.

Because the license is unknown, the images are not redistributed here. The weights of the model trained on them (`best.pt`, ConvNeXt-Tiny, seed 42) are published as an asset of the GitHub release `v1.0` so that the app runs without training; that was the author's decision, and it carries the uncertainty about the training data's license.

## Local layout (planned)

```
data/
  raw/
    kaggle_brain_tumour_classification/
      glioma/  meningioma/  notumor/  pituitary/
  interim/     split tables, hashes, duplicate reports
  processed/   final train / validation / test lists
```

Until a download script exists, download the dataset from the Kaggle page above (a Kaggle account is required) and place the class folders as shown.

## External test candidate

| Field | Value |
|:--|:--|
| Name | Brain Tumor MRI Dataset (Masoud Nickparvar) |
| Source | https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |
| License | CC BY 4.0 (attribution required) |
| Origin | Combination of figshare, SARTAJ and Br35H, according to the dataset description |
| Local copy | `data/raw/nickparvar/{Training,Testing}/` (not tracked) |

The downloaded version has 7,200 JPG images, balanced (1,400 per class in Training, 400 per class in Testing), with mixed sizes and RGB or grayscale modes.

### Overlap with the training data

Measured with the same 64-bit difference hash used for curation (Hamming distance to the nearest image of the Kaggle training dataset):

| Subset | Distance 0 | Distance <= 2 |
|:--|--:|--:|
| All (7,200) | 56.4 % | 71.0 % |
| Training (5,600) | 68.0 % | 82.9 % |
| Testing (1,600) | 15.8 % | 29.4 % |

Among the pairs at distance <= 2, 99.8 % carry the same class label, so the matches are real and not hash collisions. The Nickparvar dataset is therefore largely contained in the Kaggle dataset used for training, and cannot be used as an external test as it is.

The hash distance alone is not a reliable criterion (see `docs/curation.md`), so the external set is built with the same pixel correlation used for the groups: an image is dropped if any training image has a correlation >= 0.95 with it. Nickparvar is never used for training, so its own Training and Testing folders do not matter. This keeps 2,180 of the 7,200 images (glioma 454, meningioma 531, notumor 991, pituitary 204). Cropped, flipped or rotated copies may still remain.
