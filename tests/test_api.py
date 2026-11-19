"""The bridge used by the front end, with a fake model (no torch)."""
import base64
import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from app.api import Api
from app.backend.settings import Settings


class Fake:
    def __init__(self, probs):
        self.probs = np.array(probs)

    def predict(self, path):
        return self.probs


def png(seed=0, fmt="PNG"):
    buf = io.BytesIO()
    Image.fromarray(np.random.RandomState(seed).randint(0, 255, (64, 64), dtype=np.uint8)).save(buf, fmt)
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def api(tmp_path):
    s = Settings(data_dir=str(tmp_path / "d"))
    s.ensure_dirs()
    open(os.path.join(s.weights_dir, "v1.pt"), "wb").write(b"x")
    a = Api(s, service_factory=lambda p: Fake([0.40, 0.45, 0.10, 0.05]))
    assert a.register("alice", "correct horse")["ok"]
    assert a.login("alice", "correct horse")["ok"]
    return a


def test_login_errors_are_codes_and_calls_need_a_session(tmp_path):
    s = Settings(data_dir=str(tmp_path / "d"))
    a = Api(s)
    assert a.register("ab", "correct horse") == {"ok": False, "error": "bad_username"}
    assert a.register("alice", "short") == {"ok": False, "error": "short_password"}
    assert a.register("alice", "correct horse")["ok"]
    assert a.register("alice", "another one!!") == {"ok": False, "error": "username_taken"}
    assert a.login("alice", "wrong password") == {"ok": False, "error": "bad_credentials"}
    assert a.predict("x.png", png()) == {"ok": False, "error": "not_logged_in"}
    assert a.history()["error"] == "not_logged_in" and a.status()["error"] == "not_logged_in"
    a.login("alice", "correct horse")
    a.logout()
    assert a.queue()["error"] == "not_logged_in"


def test_predict_rejects_non_images_and_other_formats(api):
    assert api.predict("a.png", base64.b64encode(b"not an image").decode())["error"] == "not_image"
    assert api.predict("a.gif", png(fmt="GIF"))["error"] == "not_image"
    assert api.predict("a.png", "%%%")["error"] == "not_image"


def test_predict_reports_margin_and_low_margin_flag(api):
    r = api.predict("slice.png", png())
    assert r["ok"] and r["label"] == "meningioma" and r["margin"] == pytest.approx(0.05)
    assert r["low_margin"] is True and r["model_version"] == "v1" and r["image"].startswith("data:image/png;base64,")
    api._settings.margin_threshold = 0.01
    assert api.predict("slice.png", png())["low_margin"] is False


def test_save_then_review_flow_and_duplicates(api):
    r = api.predict("slice_a.png", png(1))
    assert api.save(r["md5"], True)["ok"]
    assert api.save(r["md5"], True) == {"ok": False, "error": "duplicate"}
    assert api.predict("slice_a_again.png", png(1))["duplicate"] is True
    st = api.status()
    assert st["pending"] == 1 and st["badge"] == 1 and st["confirmed"] == 0
    q = api.queue()["items"]
    assert [i["name"] for i in q] == ["slice_a.png"]
    assert api.image(q[0]["id"])["image"].startswith("data:image/png")
    assert api.review(q[0]["id"], "glioma", "looks like a glioma")["ok"]
    assert api.review(q[0]["id"], "glioma") == {"ok": False, "error": "not_pending"}
    assert api.review(q[0]["id"], "banana")["error"] == "bad_class"
    h = api.history()["rows"]
    assert h[0]["status"] == "corrected" and h[0]["true_class"] == "glioma" and api.status()["confirmed"] == 1


def test_queue_is_sorted_by_lowest_margin(api):
    api._factory = lambda p: Fake([0.3, 0.4, 0.2, 0.1])
    api._service = None
    a = api.predict("hi.png", png(1))
    api.save(a["md5"], True)
    api._factory = lambda p: Fake([0.34, 0.35, 0.2, 0.11])
    api._service, api._service_version = None, None
    b = api.predict("lo.png", png(2))
    api.save(b["md5"], True)
    assert [i["name"] for i in api.queue()["items"]] == ["lo.png", "hi.png"]


def test_users_do_not_see_each_others_images(api):
    r = api.predict("mine.png", png(3))
    api.save(r["md5"], True)
    pid = api.queue()["items"][0]["id"]
    api.logout()
    api.register("bob", "correct horse")
    api.login("bob", "correct horse")
    assert api.history()["rows"] == [] and api.queue()["items"] == []
    assert api.image(pid)["error"] == "missing_image"
    assert api.review(pid, "glioma")["error"] == "not_pending"
    assert api.delete(pid)["error"] == "in_training"


def test_history_filters_search_by_name_and_csv(api):
    for i, n in enumerate(["alpha.png", "beta.png"]):
        r = api.predict(n, png(10 + i))
        api.save(r["md5"], False)
    assert [x["name"] for x in api.history(query="ALP")["rows"]] == ["alpha.png"]
    assert len(api.history(cls="meningioma")["rows"]) == 2 and api.history(cls="glioma")["rows"] == []
    assert api.history(status="saved")["rows"][0]["status"] == "saved"
    csv_text = api.export_csv()["csv"]
    assert csv_text.splitlines()[0].startswith("date,image,predicted") and "alpha.png" in csv_text


def test_retrain_needs_enough_confirmed_images(api):
    assert api.retrain() == {"ok": False, "error": "not_enough"}


def test_missing_weights_is_reported(tmp_path):
    s = Settings(data_dir=str(tmp_path / "d"))
    a = Api(s, service_factory=lambda p: Fake([1, 0, 0, 0]))
    a.register("alice", "correct horse")
    a.login("alice", "correct horse")
    assert a.predict("a.png", png())["error"] == "no_weights"
    assert a.status()["weights_installed"] is False
