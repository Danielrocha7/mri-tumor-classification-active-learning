# Data

Data files are **not** stored in Git (the `data/` folder is in `.gitignore`). This file records where they come from, what was checked and where they are kept locally.

## Sources

| Source | Version / date checked | License | Notes |
|:--|:--|:--|:--|
| Kaggle, [Brain Tumour Classification](https://www.kaggle.com/datasets/rishiksaisanthosh/brain-tumour-classification) (uploader: Rishik Sai Santhosh) | Page checked on 2026-11-17; shown as updated about 2 years earlier | "Unknown" on the Kaggle page | 4 classes: glioma, meningioma, pituitary, no tumor. The page states 18,279 images in JPG format and gives no origin for them. |

## To reconcile

- The folder used by the original project holds 22,017 PNG images (5,284 glioma, 5,332 meningioma, 5,621 no tumor, 5,780 pituitary), not 18,279 JPG images, so it is not known to match the current Kaggle version.
- The page does not say where the images come from. Public brain MRI datasets on Kaggle are often combinations of earlier ones, which can put the same image on both sides of a split. A first check found no byte-identical files but many near-duplicates between the original training and test sets (see `docs/evaluation_audit.md`), so the near-duplicate groups must be kept together in any new split.

Because the license is unknown, the images are not redistributed here, and models trained on them are not published either.

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
