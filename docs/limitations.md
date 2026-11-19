# Limitations and possible improvements

Running list, updated as the work goes on. It becomes the limitations section of the README at the end. The original work was a first-year master's group project; several limits below come from that starting point.

## From the original work

- Evaluation: test set used to choose the checkpoint, no validation set, near-duplicates across partitions, index mismatch between the saved labelled set and the full folder. See `evaluation_audit.md`.
- Data: licence and provenance of the dataset are unknown, no patient identifiers, no scanner or sequence metadata.

## Of this project (to fill in)

- Near-duplicate detection (pixel correlation of 64x64 thumbnails) is not invariant to flips, rotations or crops, so some duplicates may remain.
- Patient-level splitting is impossible without patient IDs; group-level splitting is only a proxy.
- Simulated active learning uses labels that already exist; it does not measure annotation effort or annotator error.
- No prospective or clinical validation. This is not a clinical decision tool.
- No fully independent external set: the one used (Nickparvar, restricted to images without a near-twin in the training data) shares sources with the training data. BRISC was considered and not used because it is a curated subset of the same Nickparvar dataset (see `decisions.md`).
- Active learning results rest on three seeds, one backbone and one training recipe; the better strategies are not separated, and the ranking could change with another setup. Test and external probabilities exist at four budgets only. See `active_learning_results.md`.
- The checkpoint of every active-learning round is chosen on a 400-image validation subset, which is noisy at the small differences between strategies.
- The original-protocol arm differs from the random baseline in model, training and starting set, so its comparison with the other arms does not isolate the query strategy.
