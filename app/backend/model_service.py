"""Inference with the supervised ConvNeXt-Tiny (scripts/07_train_supervised.py, best-validation epoch).

The preprocessing must be identical to training: grayscale, 224x224 bilinear, ImageNet statistics on three
copies of the channel. torch is imported on first use so that the rest of the backend (and its tests) run
without it."""
import threading

import numpy as np
from PIL import Image

from .db import CLASSES

SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_image(path):
    """uint8 array (224, 224): grayscale and bilinear resize, as in scripts/06_pack_images.py."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)


def build_model():
    """Same architecture as the training script, without downloading pretrained weights (they come from best.pt)."""
    import torch.nn as nn
    from torchvision import models
    m = models.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(CLASSES))
    return m


def describe(probs):
    """Prediction summary from the 4 probabilities. 'confidence' is a softmax output, not a calibrated probability."""
    probs = np.asarray(probs, dtype=float)
    top2 = np.sort(probs)[-2:]
    return {"label": CLASSES[int(probs.argmax())], "confidence": float(top2[1]), "margin": float(top2[1] - top2[0]),
            "probs": {c: float(p) for c, p in zip(CLASSES, probs)}}


class ModelService:
    def __init__(self, weights_path, device="cpu"):
        import torch
        self._torch = torch
        self.device = torch.device(device)
        self.weights_path = weights_path
        self.model = build_model()
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()
        self._mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)
        self._lock = threading.Lock()  # one forward pass at a time (the UI thread and a retraining thread may overlap)

    def _batch(self, arrays):
        t = self._torch.from_numpy(np.stack(arrays)).to(self.device).float().div(255).unsqueeze(1)
        return (t.repeat(1, 3, 1, 1) - self._mean) / self._std

    def predict_arrays(self, arrays, batch=32):
        out = []
        with self._lock, self._torch.no_grad():
            self.model.eval()
            for s in range(0, len(arrays), batch):
                logits = self.model(self._batch(arrays[s:s + batch]))
                out.append(self._torch.softmax(logits.float(), 1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, path):
        """Probabilities (4,) in the order of db.CLASSES for one image file."""
        return self.predict_arrays([load_image(path)])[0]
