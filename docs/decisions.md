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
| 2026-11-17 | External test set = only Nickparvar images with no near-twin in the training data (distance > 2) | 71 % of the Nickparvar images have a near-twin in the Kaggle dataset, so the full dataset is not external | `data/README.md` | Adopted |
