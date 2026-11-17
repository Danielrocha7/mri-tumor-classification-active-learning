# Decision log

| Date | Decision | Reason | Evidence | Status |
|:--|:--|:--|:--|:--|
| 2026-11-17 | New public repository with a fresh history, not a fork of the private group repository | The old repository contains student numbers, a user database and data files | Pre-publication check | Planned |
| 2026-11-17 | Original group code kept in `original/`; new work lives outside it | Makes clear what is group work and what is new | `README.md` | Planned |
| 2026-11-17 | Images, model weights, databases and run-time files are not tracked | The dataset license is "Unknown" on Kaggle; the database holds user accounts | `data/README.md`, `.gitignore` | Adopted |
| 2026-11-17 | README and documentation in English | Portfolio audience | - | Adopted |
| 2026-11-17 | Scripts: short header, flat structure, short comments; results confirmed identical after any rewrite of an original script | Easier to read and to compare with the originals | - | Adopted |
| 2026-11-17 | Code license: MIT | Permissive and common for portfolio code | Needs the written agreement of the other two original authors | Pending |
