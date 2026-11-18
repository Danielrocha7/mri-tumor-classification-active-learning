# Active learning: design (phase 5)

Status: proposal, nothing has been run. Written before any active-learning result exists, so the choices below cannot depend on one.

## Question

With a small labelling budget, does choosing which images to label beat choosing them at random, and how close does it get to the model trained on the whole labelled pool? The original project reported 94.75% with 596 labelled images, but that number cannot be used as a reference: 129 of those 596 images are test images (see `evaluation_audit.md`, finding 3), and the labelled set ended up with 6% pituitary against 26% in the data (finding 5). The comparison here is therefore with our own random baseline and with the full-data models, not with that figure.

## Setup

- Pool: the training folds of the group-aware split (folds 1 to 5 of trainval, about 15.7k images). Test fold and external set are never touched by the loop.
- Validation: a small fixed labelled subset of fold 0 (proposed 400 images, drawn at random, same for all strategies and seeds). It is used only to choose the epoch of each round. Its size is part of the design because labelled validation data is also annotation; with the whole fold 0 (3,045 images) the low-label setting would be unrealistic.
- Model: ConvNeXt-Tiny, ImageNet-pretrained, same recipe as the baselines (AdamW, lr 3e-4, 12 epochs, same augmentation). A fresh model is trained from the pretrained weights in every round, not continued from the previous round.
- Start: 60 random images, then rounds of 60 up to 600 labelled images, then rounds of 120 up to 1,200. The point at about 600 is where the original project stopped.
- Strategies: random, entropy, margin, MC-dropout (BALD), Core-Set (k-center greedy on penultimate features), BADGE (gradient embeddings of the last layer). Uncertainty and features are computed on non-augmented images.
- Seeds: the initial set and the validation subset depend on the seed and are shared by all strategies of that seed, so the comparison is paired. Pilot with 1 seed to time it, then 3 seeds, more only if the differences need them.

## Points that need a decision

- ConvNeXt-Tiny has no dropout in its head, so MC-dropout needs a dropout layer added before the last linear layer. That changes the model compared with the baselines for that strategy only. The alternative is to run MC-dropout on the EfficientNet-B0 variant, which already has dropout, and then it is not comparable with the others. Proposed: add the dropout layer to ConvNeXt-Tiny for all strategies, so every strategy uses the same model.
- Class balance of the queried set is reported for every strategy (the original skewed it) but not enforced.

## What is reported

- Learning curves of macro-F1 against labelled images, with the mean and the spread over seeds. Recall per class at 600 and at 1,200 images (pituitary and glioma are the weak classes).
- Area under the learning curve and the difference to random, paired by seed.
- Fraction of each class in the labelled set at each budget.
- Internal test and external set are evaluated once, after the design is frozen, for the final labelled sets (retraining from scratch on each final set with the baseline recipe), next to the full-data models and the random baseline. During development only the validation subset is looked at.

## Cost

Rough estimate before measuring: a round is a training from scratch on at most 1,200 images plus one scoring pass over about 15.7k pool images (MC-dropout needs several passes). With 15 rounds, 6 strategies and 3 seeds that is 270 rounds. The pilot with one seed measures the real time before committing the rest of the free Kaggle quota.

## Limits to state in the write-up

- Labels come from an already labelled dataset, so annotation is simulated: no annotator noise and no cost differences between images.
- The near-duplicate structure of the data means a queried image can have a near-twin already labelled; this wastes budget but does not leak into test, because the test fold is group-separated.
- Three seeds may not separate strategies whose differences are small.
