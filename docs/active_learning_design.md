# Active learning: plan for phase 5

Status: frozen before any active-learning result exists, so none of the choices below can depend on one. Script: `scripts/13_active_learning.py`.

## Question

With a small labelling budget, does choosing which images to label beat choosing them at random, and how close does it get to the model trained on the whole pool? The original project reported 94.75% with 596 labelled images, but that number cannot be a reference: 129 of those 596 images are test images (`evaluation_audit.md`, finding 3), the labelled set ended with 6% pituitary against 26% in the data (finding 5), and the loop stopped early on test accuracy. The comparison here is with a random baseline run under the same conditions, with the full-pool model, and with a replica of the original protocol on the leak-free splits.

## What the original loop did (read from the original code)

Class-balanced seed of 1% of the training split (about 176 images), 5 epochs, then up to 20 iterations: score the pool by prediction entropy, add the 20 most uncertain images, fine-tune the same model for 3 epochs (Adam, lr 1e-4, batch 32, EfficientNet-B0 with a partly frozen backbone and a new head). Early stopping with patience 5 on test accuracy.

## Setup

- Pool: the training folds 1 to 5 of the group-aware split (15,717 images). The test fold and the external set are never read by the loop.
- Validation: 400 images drawn at random from fold 0, per seed. Used only to choose the checkpoint inside a round and to draw the learning curves during development. It is kept small because labelled validation data is also annotation; the whole fold (3,045 images) would make the low-label setting unrealistic.
- Main model: ConvNeXt-Tiny, ImageNet-pretrained, with a dropout layer (p = 0.3) before the final linear layer, for every strategy (needed by MC-dropout; the baselines of phase 4 have no such layer). Every round trains a fresh model from the pretrained weights: 400 optimisation steps, batch 32, AdamW (lr 3e-4, weight decay 1e-4), cosine decay, the augmentation of the baselines; the validation subset is scored every 40 steps and the best checkpoint is kept.
- Budgets: 60 random images at the start, then rounds of 60 up to 600 labelled images, then rounds of 120 up to 1,200 (15 points on the curve). The point at about 600 is where the original stopped.
- Strategies: random, entropy, margin, MC-dropout (BALD, 20 stochastic passes through the dropout head), Core-Set (k-center greedy on the 768-d penultimate features), BADGE (k-means++ seeding on last-layer gradient embeddings). Scores and features are computed on non-augmented images.
- Original-protocol arm: the same loop as the original on the leak-free pool and with the validation subset in place of the test set: EfficientNet-B0 with the original head, a class-balanced seed of 1% of the pool (157 images), 5 epochs, then 21 iterations of 20 entropy-selected images with 3 epochs of continued fine-tuning (Adam, lr 1e-4, batch 32), no early stopping. Entropy is scored on non-augmented images, which the original may not have done (not verified).
- Full-pool reference: the same model trained on the whole pool (6 epochs), one run per seed.
- Seeds: 42, 43 and 44. The initial set and the validation subset depend on the seed only, so all strategies of a seed start from the same 60 images and see the same validation images (paired comparison).
- The test fold and the external set are predicted at 60, 300, 600 and 1,200 labelled images and the probabilities are saved, but they are not looked at while the design can still change. After the first full results are in, nothing in this document is changed; any later change is listed in `decisions.md` with its reason.

## What is reported

- Learning curves: macro-F1 against labelled images, per strategy, mean and per-seed values. Recall per class at 600 and 1,200 images (glioma and pituitary are the weak classes).
- Area under the learning curve and the difference to random, per seed (three seeds are too few for a p-value; the per-seed differences are shown).
- Class mix of the labelled set at each budget, per strategy.
- Internal test and external set, from the saved probabilities of the rounds at 600 and 1,200 images, next to the full-pool model and to the random baseline.
- The original-protocol arm next to the main arms and against the 94.75% of the original, with the reasons that number is not comparable.

## Cost

About 100 s per round (training, scoring about 15.7k pool images, validation) on a T4, so about 25 minutes per strategy and seed, about 7.5 hours for six strategies and three seeds, run as two parallel streams on the two T4 of a Kaggle session. The original-protocol arm and the full-pool reference are small. A pilot with one seed measures the real time first.

## Limits to state in the write-up

- Labels come from an existing dataset, so annotation is simulated: no annotator noise and no cost differences between images.
- Near-duplicate structure: a queried image can have a near-twin already labelled, which wastes budget. It does not leak into the test fold, which is group-separated.
- The dropout layer makes the main model differ from the phase 4 baselines; the full-pool reference uses it too.
- Three seeds may not separate strategies whose differences are small.
