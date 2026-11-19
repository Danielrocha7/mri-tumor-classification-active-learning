"""Runs the app as a local web server and opens it in the browser. Same front end and same Api as app/main.py, but
no native window, so it needs no GUI toolkit (useful in WSL, or where Windows blocks the pywebview dependencies).

Usage: python app/serve.py [--port 8765] [--no-browser]

Safety of a local server: it listens on 127.0.0.1 only, accepts only Host headers of localhost, and every API call must
carry a random token that is embedded in the page it serves (a web page from another site cannot read it)."""
import argparse
import json
import mimetypes
import os
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from app.envfile import load_env_file  # noqa: E402

FRONTEND = os.path.join(HERE, "frontend")
MAX_BODY = 40 * 1024 * 1024


def public_methods(api):
    return {n for n in dir(api) if not n.startswith("_") and callable(getattr(api, n))}


def make_server(api, port=0, token=None):
    """Returns (server, token). port 0 picks a free port."""
    token = token or secrets.token_urlsafe(24)
    allowed = public_methods(api)
    lock = threading.Lock()  # the Api keeps per-session state, so calls are handled one at a time

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _host_ok(self):
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in ("127.0.0.1", "localhost", "::1")

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, {"error": "forbidden"})
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                html = open(os.path.join(FRONTEND, "index.html"), encoding="utf-8").read()
                html = html.replace("<head>", '<head>\n<meta name="app-token" content="%s">' % token, 1)
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            full = os.path.normpath(os.path.join(FRONTEND, path.lstrip("/")))
            if not full.startswith(FRONTEND + os.sep) or not os.path.isfile(full):
                return self._send(404, {"error": "not found"})
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            return self._send(200, open(full, "rb").read(), ctype + ("; charset=utf-8" if ctype.startswith("text/") or "javascript" in ctype else ""))

        def do_POST(self):
            if not self._host_ok() or not secrets.compare_digest(self.headers.get("X-App-Token", ""), token):
                return self._send(403, {"error": "forbidden"})
            name = self.path[len("/api/"):] if self.path.startswith("/api/") else ""
            if name == "ping":
                return self._send(200, {"ok": True})
            if name not in allowed:
                return self._send(404, {"error": "unknown method"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    return self._send(413, {"ok": False, "error": "too_large"})
                args = json.loads(self.rfile.read(n) or b"{}").get("args", [])
                with lock:
                    result = getattr(api, name)(*args)
                return self._send(200, result)
            except Exception as e:  # reported to the page; the details stay in the terminal
                print("api error in %s: %r" % (name, e), file=sys.stderr)
                return self._send(500, {"ok": False, "error": "error"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server, token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    load_env_file(os.path.join(os.path.dirname(HERE), ".env"))
    from app.api import Api
    from app.backend import trainer
    from app.backend.settings import Settings
    settings = Settings()
    settings.blocked_manifest = os.environ.get("MRI_APP_MANIFEST") or os.path.join(HERE, "..", "splits", "kaggle_splits_seed42.tsv")
    settings.ensure_dirs()
    best = os.environ.get("MRI_APP_BEST_PT")
    if best and os.path.exists(best) and trainer.latest(settings)[1] is None:
        trainer.install_initial(settings, best)
    if trainer.latest(settings)[1] is None:
        print("No model weights yet. Run `python -m app.weights` to download the published weights, "
              "or set MRI_APP_BEST_PT to your own best.pt (see app/README.md).", file=sys.stderr)
    server, _ = make_server(Api(settings), a.port)
    url = "http://127.0.0.1:%d/" % server.server_address[1]
    print("Brain MRI Classifier running at", url, "(Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
