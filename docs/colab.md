# Running the baselines on Colab

Training needs a GPU, so it runs on Google Colab (free T4 is enough).

## Files to upload to Google Drive

Create a folder `mri-project` in My Drive with this structure:

```
mri-project/
  kaggle224_seed42.npz            (from data/processed, 1.1 GB)
  scripts/07_train_supervised.py
  src/utils.py
```

## Colab cells

Runtime, Change runtime type, T4 GPU. Then:

```python
from google.colab import drive
drive.mount('/content/drive')
P = '/content/drive/MyDrive/mri-project'
```

Smoke test (1 epoch on a few images, no downloads of weights):

```python
!python {P}/scripts/07_train_supervised.py --data {P}/kaggle224_seed42.npz --model resnet50 --quick --no-pretrained --out {P}/results/smoke
```

One real run (validation fold 0, test not used):

```python
!python {P}/scripts/07_train_supervised.py --data {P}/kaggle224_seed42.npz --model efficientnet_b0 --val-fold 0 --out {P}/results/efficientnet_b0_f0
```

Results (`metrics.json`, validation probabilities) are written to `results/` in Drive. The test set is only used with `--predict-test`, once, for the final model of each architecture.
