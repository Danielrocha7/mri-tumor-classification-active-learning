import numpy as np
import pytest
from PIL import Image

from app.backend import model_service as ms
from app.backend.db import CLASSES


def test_load_image_is_grayscale_224_uint8(tmp_path):
    rgb = np.random.RandomState(0).randint(0, 255, (300, 180, 3), dtype=np.uint8)
    p = tmp_path / "a.jpg"
    Image.fromarray(rgb).save(p)
    a = ms.load_image(p)
    assert a.shape == (224, 224) and a.dtype == np.uint8


def test_describe_margin_is_top1_minus_top2():
    r = ms.describe([0.1, 0.6, 0.25, 0.05])
    assert r["label"] == "meningioma" and r["margin"] == pytest.approx(0.35) and r["confidence"] == pytest.approx(0.6)
    assert set(r["probs"]) == set(CLASSES)


def test_service_loads_weights_and_predicts(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = ms.build_model()
    w = tmp_path / "best.pt"
    torch.save(model.state_dict(), w)
    img = tmp_path / "x.png"
    Image.fromarray(np.random.RandomState(1).randint(0, 255, (224, 224), dtype=np.uint8)).save(img)
    svc = ms.ModelService(str(w))
    p = svc.predict(str(img))
    assert p.shape == (4,) and p.sum() == pytest.approx(1.0, abs=1e-5)
    # same weights, same image -> same output as calling the model directly with the training preprocessing
    x = torch.from_numpy(ms.load_image(img)).float().div(255).view(1, 1, 224, 224).repeat(1, 3, 1, 1)
    x = (x - torch.tensor(ms.MEAN).view(1, 3, 1, 1)) / torch.tensor(ms.STD).view(1, 3, 1, 1)
    model.eval()
    with torch.no_grad():
        ref = torch.softmax(model(x), 1)[0].numpy()
    assert np.allclose(p, ref, atol=1e-5)
    # a batch gives the same result as single predictions
    b = svc.predict_arrays([ms.load_image(img)] * 3)
    assert b.shape == (3, 4) and np.allclose(b[0], p, atol=1e-5)


def test_service_rejects_wrong_weights(tmp_path):
    torch = pytest.importorskip("torch")
    bad = tmp_path / "bad.pt"
    torch.save({"foo": torch.zeros(1)}, bad)
    with pytest.raises(Exception):
        ms.ModelService(str(bad))
