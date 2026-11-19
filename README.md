# Brain MRI Tumor Classification with Active Learning

Four-class classification of brain MRI slices (glioma, meningioma, pituitary tumor, no tumor) with a convolutional network, an active learning loop, and a small desktop application where a user confirms predictions that are later used to retrain the model.

This repository continues a group project from the Human-Centric Artificial Intelligence course (University of Coimbra, 2025). The original code is not included in this repository and is credited [below](#credits). Everything here is new work by Daniel Rocha, aimed at a sounder evaluation and a reproducible codebase.

> **Status (2026-11-19):** finished portfolio project. Leak-free splits, baselines, internal and external evaluation, an active-learning study and a small application are done; see [Results](#results), [Application](#application) and [Limitations](#limitations).

> **Not a clinical tool.** This is a learning and portfolio project. It must not be used to diagnose, triage or make any medical decision.

## Contents

[Original work and what is new](#original-work-and-what-is-new) · [Repository structure](#repository-structure) · [Application](#application) · [Environment](#environment) · [Data](#data) · [Limitations](#limitations) · [Results](#results) · [Reproducibility](#reproducibility) · [License](#license) · [Credits](#credits)

## Original work and what is new

The original project (group work, 2025) did the following:

- A 4-class classifier: EfficientNet-B0 pretrained on ImageNet with a new classification head, 224x224 inputs.
- A simulated active learning loop: a fixed test set of 4,404 images is set aside, the training split (17,613 images) starts with 1% labelled and the rest is treated as unlabelled. Each iteration selects the 20 images with the highest prediction entropy and fine-tunes for 3 epochs. The saved final state has 596 labelled images and 17,017 still in the pool.
- A desktop application (customtkinter) with user accounts, predictions with a confidence value, a history, and a confirmation step. After 20 confirmed cases the model is retrained.

## Audit of the original evaluation

A first audit of the original project (2026-11-17, details and method in [`docs/evaluation_audit.md`](docs/evaluation_audit.md)) found that its reported accuracy cannot be taken at face value:

- About a quarter of the test images have a training image with an identical perceptual hash, and about half are within a Hamming distance of 2. There are no byte-identical copies, but the images are near-duplicates.
- The best checkpoint and the early stopping were chosen on the test set.
- The labelled-set table (`treino_rotulado.csv`) and the retraining script read saved indices against the wrong list: 129 of the 596 labelled images are test images.

What this repository changes is tracked in [`docs/decisions.md`](docs/decisions.md). The original code is not republished here; the audit of what it did is in [`docs/evaluation_audit.md`](docs/evaluation_audit.md).

## Repository structure

```
mri-tumor-classification-active-learning/
├── README.md
├── environment.yml    # conda environment (draft, not tested yet)
├── .gitignore
├── configs/           # seeds and parameters (config.yaml)
├── data/              # NOT tracked by Git (see data/README.md)
├── docs/              # decision log and evaluation notes
├── scripts/           # numbered, flat scripts that run the pipeline step by step
├── src/               # shared helpers
├── app/               # web interface in a window or in the browser, with its Python backend (app/README.md)
├── results/           # light outputs: supervised runs and active-learning predictions
└── tests/             # checks for splits, active-learning isolation, the app backend and the weights download
```

## Application

A small app around the model: sign in, classify a slice, send the uncertain ones to a review queue, confirm the correct class, and fine-tune the model on the confirmed images. Details, limits and how to run it are in [`app/README.md`](app/README.md).

![Prediction view: one slice, the four class scores and the margin](docs/screens/prediction.png)

![Model support view: the review queue, lowest margin first, with the retraining status](docs/screens/support.png)

![History view: saved predictions with status and model version](docs/screens/history.png)

The screenshots show four images from the public dataset's test folder (one per class), only displayed. Nothing from the test folder is used for training. The margin is a weak error detector (most of the model's errors have a high margin), so the review queue is a help, not a safety net. The app is a research prototype.

## Environment

```bash
conda env create -f environment.yml
conda activate mri-tumor-classification-active-learning
```

The environment file is a draft and has not been tested yet. Exact versions will be pinned once the pipeline runs.

## Data

No image is stored in this repository. Where every piece of data comes from:

| Role | Source | License | Used for | Redistributed here? |
|:--|:--|:--|:--|:--|
| Training, validation and internal test | Kaggle, [Brain Tumour Classification](https://www.kaggle.com/datasets/rishiksaisanthosh/brain-tumour-classification) (uploader: Rishik Sai Santhosh) | "Unknown" on the Kaggle page; the page gives no origin for the images | Supervised models, active learning, the internal test | No. Only the split tables (`splits/kaggle_splits_seed42.tsv`: file names, md5, groups, folds) |
| External evaluation | Kaggle, [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Masoud Nickparvar), a combination of figshare, SARTAJ and Br35H according to its description | CC BY 4.0 | External evaluation on the 2,180 images with no near-twin in the training data (`splits/external_nickparvar_c095_manifest.tsv`) | No. Four slices appear, scaled, in the screenshots, with attribution (see [Credits](#credits)) |
| Pretrained backbones | ImageNet-pretrained weights from `torchvision` (ConvNeXt-Tiny and the other CNNs); DINOv2-B from `torch.hub` for one baseline | The terms of those projects | Starting point of the models | No (downloaded by the code) |
| Weights of the trained model | Trained here on the first dataset | The training data's license is unknown | The app | Yes, as the asset `best.pt` of the release `v1.0` (see [Reproducibility](#reproducibility)) |

The first dataset may combine earlier public datasets, which can put near-identical images on both sides of a split, so the splits group near-duplicates (`data/README.md`). The folder used by the original project (22,017 PNG images) is not known to match the Kaggle version (18,279 JPG images). Source, version, checks and the expected local layout are in [`data/README.md`](data/README.md). Nothing here is a clinical result.

## Limitations

The full list, with the reasoning, is in [`docs/limitations.md`](docs/limitations.md). The ones that matter most when reading the numbers:

- The test and external scores are very high and the external set shares sources with the training data, so they say little about performance on a new scanner or hospital.
- Group-aware splits are a proxy: without patient identifiers, images of one patient can still sit on both sides of a split.
- The active-learning study is simulated (the labels already exist), uses three seeds and one backbone, and does not separate the better strategies.
- In the app, the margin is a weak error detector and softmax outputs are not calibrated; the retraining check only guards against forgetting, not against poor generalisation.
- Not a clinical tool. No prospective or clinical validation.

## Results

**Supervised baselines (validation, 6-fold group-aware cross-validation).** Mean of the best-epoch validation macro-F1 per fold; details and caveats in [`docs/baselines.md`](docs/baselines.md).

| Model | Mean | SD |
|:--|:--|:--|
| ConvNeXt-Tiny | 0.9970 | 0.0013 |
| EfficientNet-B0 | 0.9944 | 0.0026 |
| ResNet-50 | 0.9940 | 0.0030 |
| EfficientNet-B0 (head of the original project) | 0.9914 | 0.0020 |
| DINOv2-B, frozen, linear layer only (not in the figure) | 0.9192 | 0.0081 |

![Validation macro-F1 of the baselines](docs/figures/baselines_validation.png)

The differences are small and the folds share training data, so no significance test is reported. For reference, the original project reported a best test accuracy of 0.9475, where the best checkpoint was chosen on the test set itself, so that figure is likely optimistic.

**Closed test set (2,989 images, group-aware split, opened once per final model).** Three seeds per model; macro-F1 mean (SD); details, intervals and limits in [`docs/test_evaluation.md`](docs/test_evaluation.md).

| Model | Test macro-F1 | Test accuracy per seed |
|:--|:--|:--|
| ConvNeXt-Tiny | 0.9965 (0.0042) | 0.9987, 0.9990, 0.9916 |
| EfficientNet-B0 (original head) | 0.9927 (0.0009) | 0.9936, 0.9920, 0.9923 |

ConvNeXt-Tiny is better in two seeds and equal in one, so the ranking is not firm. This is an internal test on the same dataset.

**External set (2,180 Nickparvar images with no near-twin in the training data).** Same models and seeds; details and limits in [`docs/external_evaluation.md`](docs/external_evaluation.md).

| Model | Internal test macro-F1 | External macro-F1 (SD) | Glioma recall, external |
|:--|:--|:--|:--|
| ConvNeXt-Tiny | 0.9965 | 0.9495 (0.0080) | 0.859 |
| EfficientNet-B0 (original head) | 0.9927 | 0.9257 (0.0090) | 0.854 |

Performance drops on another source, mostly because about 14% of the gliomas are missed by both architectures; ConvNeXt-Tiny stays ahead in all three seeds. The glioma errors are described in [`docs/external_evaluation.md`](docs/external_evaluation.md); no cause is established.

![Internal test against external set: macro-F1 per seed and recall per class](docs/figures/internal_vs_external.png)

![External confusion matrices](docs/figures/external_confusion.png)

**Active learning (6 strategies and random sampling, ConvNeXt-Tiny, 60 to 1,200 labelled images, 3 seeds).** Test macro-F1 at 1,200 labelled images (7.6% of the pool), with the full-pool reference trained with the same short recipe; details, tables for all budgets and limits in [`docs/active_learning_results.md`](docs/active_learning_results.md).

| Strategy | Internal test | External set |
|:--|:--|:--|
| Random | 0.960 | 0.866 |
| Entropy | 0.988 | 0.940 |
| Margin | 0.991 | 0.941 |
| MC-dropout | 0.988 | 0.945 |
| Core-Set | 0.980 | 0.915 |
| BADGE | 0.991 | 0.945 |
| Full pool (15,717 images, 6 epochs) | 0.992 | 0.943 |

Every strategy beat random sampling in all three seeds from 600 images on. The four best strategies are not separated with three seeds, and the external glioma recall stays near 0.85 whatever the strategy. The replica of the original loop (EfficientNet-B0, 577 labelled images) scores about the same as random sampling at that size, so the accuracy reported by the original project says little about the query strategy.

![Active learning curves](docs/figures/al_curves.png)

## Reproducibility

- **Seeds:** fixed through the value in `configs/config.yaml`.
- **Versions:** `environment.yml`. The analysis libraries are pinned to the versions used here; torch and torchvision are pinned to the Kaggle versions read on 2026-11-19, a few days after the runs. The file has not been built with conda yet.
- **Active learning, step by step** (the training runs need a GPU; they were run on a Kaggle T4, 22 to 32 min per strategy and seed):
  1. Pack the images with `scripts/06_pack_images.py` (Kaggle set and the external set, see `data/README.md`).
  2. `python scripts/13_active_learning.py --help` lists the arms (random, entropy, margin, mcdropout, coreset, badge, original, full), the seed and the paths; a run writes its folder under the output directory and can be started again, it resumes from the saved labelled order.
  3. `python scripts/15_make_labels.py splits/kaggle_splits_seed42.tsv splits/external_nickparvar_c095_manifest.tsv labels.npz` writes the labels file (identical to `results/active_learning/labels.npz`).
  4. `python scripts/14_active_learning_analysis.py results/active_learning results/active_learning/labels.npz docs/figures` recomputes the figure and the tables without images or a GPU.
- **Weights for the app:** asset `best.pt` of the release `v1.0` (SHA-256 `33e280ce1dcba71a9111743b1905ab7b8838dcbf436912edfa8090c553d7c39d`). `python -m app.weights` downloads it, checks the hash and installs it; see `app/README.md`. The training data's license is unknown (`data/README.md`).
- **Tests:** `pytest tests/`. `test_active_learning_isolation.py` checks that every labelled set comes from the pool only (never from the validation fold or the test fold); `test_splits.py` needs the local curation files and is skipped without them.
- **Data:** source and checks in `data/README.md`.
- **Decisions:** logged in `docs/decisions.md`.
- **Note:** small differences between machines (CPU/GPU, library versions) can remain because of non-deterministic training operations.

## License

The code is released under the [MIT license](LICENSE) (Copyright 2026 Daniel Rocha). The original group code is not part of this repository. The dataset is not covered by the license and keeps its own terms.

## Credits

Original group project, Human-Centric Artificial Intelligence, University of Coimbra, 2025: João Pedro Ferreira, João Frederico Lopes and Daniel Rocha. 

The screenshots in `docs/screens/` show four slices from the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar (CC BY 4.0; combination of figshare, SARTAJ and Br35H), scaled but not otherwise changed.

Maintained by Daniel Rocha.

This project was improved with the help of artificial intelligence (Claude, by Anthropic), used on the code, tests, interface and documentation. The author directed the work and is responsible for its content.
