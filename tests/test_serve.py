"""The local web server of app/serve.py: token and host checks, static files, and calling the Api over HTTP."""
import base64
import http.client
import io
import json
import os
import sys
import threading

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from app.api import Api
from app.backend.settings import Settings
from app.serve import make_server


class Fake:
    def predict(self, path):
        return np.array([0.1, 0.7, 0.15, 0.05])


@pytest.fixture
def srv(tmp_path):
    s = Settings(data_dir=str(tmp_path / "d"))
    s.ensure_dirs()
    open(os.path.join(s.weights_dir, "v1.pt"), "wb").write(b"x")
    server, token = make_server(Api(s, service_factory=lambda p: Fake()), 0, token="tok")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], token
    server.shutdown()


def req(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers or {})
    r = c.getresponse()
    data = r.read()
    return r.status, data


def post(port, name, args, token="tok", host=None):
    h = {"X-App-Token": token, "Content-Type": "application/json"}
    if host:
        h["Host"] = host
    return req(port, "POST", "/api/" + name, {"args": args}, h)


def test_index_carries_the_token_and_static_files_are_served(srv):
    port, _ = srv
    code, body = req(port, "GET", "/")
    assert code == 200 and b'name="app-token" content="tok"' in body
    assert req(port, "GET", "/app.js")[0] == 200
    assert req(port, "GET", "/../api.py")[0] in (403, 404)
    assert req(port, "GET", "/%2e%2e/api.py")[0] in (403, 404)


def test_api_needs_the_token_and_a_local_host(srv):
    port, _ = srv
    assert post(port, "session", [], token="wrong")[0] == 403
    assert post(port, "session", [], token="")[0] == 403
    assert post(port, "session", [], host="evil.example.com")[0] == 403
    assert req(port, "GET", "/", headers={"Host": "evil.example.com"})[0] == 403


def test_only_public_api_methods_can_be_called(srv):
    port, _ = srv
    assert post(port, "_conn", [])[0] == 404
    assert post(port, "__init__", [])[0] == 404
    assert post(port, "ping", [])[0] == 200


def test_full_flow_over_http(srv):
    port, _ = srv
    assert json.loads(post(port, "register", ["alice", "correct horse"])[1])["ok"]
    assert json.loads(post(port, "login", ["alice", "correct horse"])[1])["ok"]
    buf = io.BytesIO()
    Image.fromarray(np.random.RandomState(0).randint(0, 255, (64, 64), dtype=np.uint8)).save(buf, "PNG")
    r = json.loads(post(port, "predict", ["a.png", base64.b64encode(buf.getvalue()).decode()])[1])
    assert r["ok"] and r["label"] == "meningioma"
    assert json.loads(post(port, "save", [r["md5"], True])[1])["ok"]
    assert len(json.loads(post(port, "queue", [])[1])["items"]) == 1


def test_wrong_argument_count_is_an_error_not_a_crash(srv):
    port, _ = srv
    code, body = post(port, "login", [])
    assert code == 500 and json.loads(body)["ok"] is False
    assert post(port, "session", [])[0] == 200  # the server is still alive
