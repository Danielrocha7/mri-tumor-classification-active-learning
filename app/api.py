"""Bridge between the web front end and the backend. pywebview exposes the public methods of Api to JavaScript
(window.pywebview.api.<method>); each returns plain dicts and lists that become JSON.

Every call opens its own sqlite connection, because pywebview runs calls on different threads. Errors the user
can cause (wrong password, not an image) come back as {"ok": False, "error": <code>}; the front end translates the code."""
import base64
import csv
import hashlib
import io
import os

from PIL import Image

from .backend import auth, db, retrain, trainer
from .backend.db import CLASSES
from .backend.model_service import describe
from .backend.settings import Settings

MAX_BYTES = 25 * 1024 * 1024
THUMB = 640

_AUTH_CODES = {"Username": "bad_username", "Password: at least": "short_password", "Password: at most": "long_password",
               "That username": "username_taken"}


def _code(message):
    for prefix, code in _AUTH_CODES.items():
        if message.startswith(prefix):
            return code
    return "error"


def _data_url(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def thumbnail(path):
    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail((THUMB, THUMB))
        return _data_url(im)


class Api:
    def __init__(self, settings=None, service_factory=None):
        self._settings = settings or Settings()
        self._settings.ensure_dirs()
        self._factory = service_factory or self._default_factory
        self._service, self._service_version = None, None
        self._user_id, self._username = None, None
        self._last = {}       # md5 -> result of the latest prediction made in this session (not yet in the history)
        self._retrain = {"running": False, "last": None}

    # ----- plumbing
    @staticmethod
    def _default_factory(path):
        from .backend.model_service import ModelService
        return ModelService(path)

    def _conn(self):
        return db.connect(self._settings.db_path)

    def _model(self):
        n, path = trainer.latest(self._settings)
        if path is None:
            return None, None
        if self._service is None or self._service_version != n:
            self._service, self._service_version = self._factory(path), n
        return self._service, "v%d" % n

    def _need_login(self):
        return None if self._user_id is not None else {"ok": False, "error": "not_logged_in"}

    # ----- accounts
    def register(self, username, password):
        c = self._conn()
        try:
            ok, msg = auth.register(c, username, password)
        finally:
            c.close()
        return {"ok": True} if ok else {"ok": False, "error": _code(msg)}

    def login(self, username, password):
        c = self._conn()
        try:
            uid = auth.login(c, username, password)
        finally:
            c.close()
        if uid is None:
            return {"ok": False, "error": "bad_credentials"}
        self._user_id, self._username = uid, username
        self._last = {}
        return {"ok": True, "username": username}

    def logout(self):
        self._user_id, self._username, self._last = None, None, {}
        return {"ok": True}

    def session(self):
        return {"logged_in": self._user_id is not None, "username": self._username}

    # ----- state shown in the header and the Model support tab
    def status(self):
        if (e := self._need_login()):
            return e
        c = self._conn()
        try:
            rounds = [dict(r) for r in c.execute("SELECT * FROM retrain_rounds ORDER BY id DESC LIMIT 20").fetchall()]
            n, _ = trainer.latest(self._settings)
            ready_rows, blocked = retrain.select_training_rows(c, self._settings)
            return {"ok": True, "weights_installed": n is not None, "model_version": "v%d" % n if n else None,
                    "badge": db.count_pending(c, self._user_id, self._settings.margin_threshold),
                    "pending": db.count_pending(c, self._user_id),
                    "confirmed": len(ready_rows), "retrain_min": self._settings.retrain_min,
                    "margin_threshold": self._settings.margin_threshold, "rounds": rounds,
                    "retraining": self._retrain["running"], "last_round": self._retrain["last"],
                    "weights_dir": self._settings.weights_dir}
        finally:
            c.close()

    # ----- prediction
    def predict(self, name, b64):
        """name: file name shown to the user; b64: file content, base64 (no data: prefix)."""
        if (e := self._need_login()):
            return e
        try:
            raw = base64.b64decode(b64, validate=False)
        except Exception:
            return {"ok": False, "error": "not_image"}
        if len(raw) > MAX_BYTES:
            return {"ok": False, "error": "too_large"}
        try:
            with Image.open(io.BytesIO(raw)) as im:
                fmt = (im.format or "").upper()
                im.verify()
        except Exception:
            return {"ok": False, "error": "not_image"}
        if fmt not in ("PNG", "JPEG"):
            return {"ok": False, "error": "not_image"}
        service, version = self._model()
        if service is None:
            return {"ok": False, "error": "no_weights"}
        md5 = hashlib.md5(raw).hexdigest()
        path = os.path.join(self._settings.images_dir, md5 + (".png" if fmt == "PNG" else ".jpg"))
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(raw)
        probs = service.predict(path)
        res = describe(probs)
        c = self._conn()
        try:
            dup = db.find_duplicate(c, self._user_id, md5)
        finally:
            c.close()
        res.update(ok=True, md5=md5, name=os.path.basename(name or "image"), model_version=version,
                   low_margin=res["margin"] < self._settings.margin_threshold,
                   duplicate=bool(dup), image=thumbnail(path))
        self._last[md5] = {"path": path, "probs": res["probs"], "version": version, "name": res["name"]}
        return res

    def save(self, md5, send_to_review):
        """Store the latest prediction of this image in the history; optionally put it in the review queue."""
        if (e := self._need_login()):
            return e
        last = self._last.get(md5)
        if last is None:
            return {"ok": False, "error": "unknown_image"}
        c = self._conn()
        try:
            if db.find_duplicate(c, self._user_id, md5):
                return {"ok": False, "error": "duplicate"}
            pid = db.add_prediction(c, self._user_id, last["path"], md5, last["probs"], last["version"], bool(send_to_review))
        finally:
            c.close()
        # the file name shown in the history is the one the user chose, kept next to the stored image
        with open(os.path.join(self._settings.images_dir, "names.tsv"), "a", encoding="utf-8") as f:
            f.write("%s\t%s\n" % (md5, last["name"].replace("\t", " ").replace("\n", " ")))
        return {"ok": True, "id": pid}

    def _names(self):
        out = {}
        p = os.path.join(self._settings.images_dir, "names.tsv")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                k, _, v = line.rstrip("\n").partition("\t")
                out[k] = v
        return out

    # ----- review queue
    def queue(self):
        if (e := self._need_login()):
            return e
        names = self._names()
        c = self._conn()
        try:
            import json
            items = [{"id": r["id"], "name": names.get(r["image_md5"], os.path.basename(r["image_path"])),
                      "probs": json.loads(r["probs"]), "predicted": r["predicted"], "margin": r["margin"]}
                     for r in db.queue(c, self._user_id)]
        finally:
            c.close()
        return {"ok": True, "items": items}

    def image(self, pred_id):
        if (e := self._need_login()):
            return e
        c = self._conn()
        try:
            r = c.execute("SELECT image_path FROM predictions WHERE id = ? AND user_id = ?", (pred_id, self._user_id)).fetchone()
        finally:
            c.close()
        if r is None or not os.path.exists(r["image_path"]):
            return {"ok": False, "error": "missing_image"}
        return {"ok": True, "image": thumbnail(r["image_path"])}

    def review(self, pred_id, true_class, notes=""):
        if (e := self._need_login()):
            return e
        if true_class not in CLASSES:
            return {"ok": False, "error": "bad_class"}
        c = self._conn()
        try:
            ok = db.review(c, pred_id, self._user_id, true_class, (notes or "")[:500])
        finally:
            c.close()
        return {"ok": ok} if ok else {"ok": False, "error": "not_pending"}

    # ----- history
    def history(self, query="", cls=None, status=None):
        if (e := self._need_login()):
            return e
        names = self._names()
        c = self._conn()
        try:
            rows = [{"id": r["id"], "name": names.get(r["image_md5"], os.path.basename(r["image_path"])),
                     "date": r["created_at"], "predicted": r["predicted"], "true_class": r["true_class"],
                     "margin": r["margin"], "model": r["model_version"], "status": r["status"], "locked": r["round_id"] is not None}
                    for r in db.history(c, self._user_id, "", cls or None, status or None)]
        finally:
            c.close()
        q = (query or "").strip().lower()
        if q:  # the stored path is a hash, so the search runs on the name the user gave
            rows = [r for r in rows if q in r["name"].lower()]
        return {"ok": True, "rows": rows[:1000]}

    def send_to_review(self, pred_id):
        if (e := self._need_login()):
            return e
        c = self._conn()
        try:
            return {"ok": db.send_to_review(c, pred_id, self._user_id)}
        finally:
            c.close()

    def delete(self, pred_id):
        if (e := self._need_login()):
            return e
        c = self._conn()
        try:
            ok = db.delete_prediction(c, pred_id, self._user_id)
        finally:
            c.close()
        return {"ok": ok} if ok else {"ok": False, "error": "in_training"}

    def export_csv(self):
        r = self.history()
        if not r.get("ok"):
            return r
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["date", "image", "predicted", "true_class", "margin", "model", "status"])
        for x in r["rows"]:
            w.writerow([x["date"], x["name"], x["predicted"], x["true_class"] or "", "%.4f" % x["margin"], x["model"], x["status"]])
        return {"ok": True, "csv": buf.getvalue()}

    # ----- retraining
    def retrain(self):
        if (e := self._need_login()):
            return e
        if self._retrain["running"]:
            return {"ok": False, "error": "already_running"}
        c = self._conn()
        try:
            if not retrain.ready(c, self._settings):
                return {"ok": False, "error": "not_enough"}
        finally:
            c.close()
        self._retrain = {"running": True, "last": None}

        def done(res):
            self._retrain = {"running": False, "last": res}
            self._service = None  # reload the weights of the newest version on the next prediction

        trainer.run_round_in_background(self._settings.db_path, self._settings, on_done=done)
        return {"ok": True}
