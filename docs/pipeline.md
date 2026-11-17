# Pipeline

Working plan for the project. Each phase is ticked in the README roadmap when it is done and its outputs are committed. Methods listed here are the plan, not results.

## Phase 1. Inventory

Count files per class, check image format and size, hash every file (MD5), and compare the folder with what the source page documents. Output: a table of the dataset as it actually is.

## Phase 2. Curation

- Exact duplicates by MD5.
- Near-duplicates by a 64-bit difference hash (dHash, 9x8), checked against pixel correlation on a sample.
- Near-duplicate groups by Hamming-distance threshold and connected components (union-find). The threshold is chosen by looking at cross-class conflicts, not by convenience.
- Groups that mix classes are reviewed by hand and removed or relabelled.
- Source-shortcut checks: file size, dimensions and metadata against class.
- Output: a manifest with hash, group and class per image (no images tracked).

## Phase 3. Splits

StratifiedGroupKFold with the group as the unit, so no near-duplicate group crosses partitions. Separate train, validation and test; the test set is touched once per final model. A unit test checks that no group appears in two partitions.

## Phase 4. Baselines

Core models: EfficientNet-B0 (the original architecture, as reference), ResNet-50, ConvNeXt-Tiny, and a DINOv2-B linear probe. Optional: Swin-T, BiomedCLIP. Checkpoints are chosen on validation only. Several seeds, results reported with confidence intervals.

## Phase 5. Active learning

Strategies: random, entropy, margin, MC-dropout, Core-Set, BADGE. Uncertainty is scored on non-augmented images. Compare a cold start with diversity sampling against random. Learning curves are repeated over seeds; the labelled pool is built from training partitions only.

## Phase 6. Evaluation

Per-class precision, recall and F1, macro-F1, confusion matrices, bootstrap confidence intervals, McNemar tests between models, calibration (ECE, Brier, temperature scaling), risk-coverage curves, quantified Grad-CAM, and external validation on a second dataset with a clear licence.

## Phase 7. Application

Redesigned interface for the retraining loop: upload, prediction with uncertainty, confirmation of labels, retraining after a batch. Credentials hashed, no plaintext passwords, no weights or database tracked.

## Compute

Curation runs locally. GPU training runs on Colab or Kaggle.
