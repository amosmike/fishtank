"""Local dashboard: `python -m idea_lab serve`, then open http://127.0.0.1:8765

Watch a run move ideas through the pipeline: node status and usage, ideas changing stage on a board,
the live log, and each idea's full history. Two datasets:

- demo: a scratch copy of sample_data/ (in .state/demo-data) run with simulated agents. No network,
  no tokens. "Reset demo" restores the copy.
- real: your data/ folder, run through the real backend after a dry-run estimate and a confirm.

Runs are subprocesses of `python -m idea_lab run`, so Stop is the same as Ctrl-C (the run is marked
stopped and can be resumed). The server binds to 127.0.0.1 only, and every POST needs an
X-Fishtank header (browsers won't send a custom header cross-site without a CORS preflight, which
this server never grants).
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import config, decisions
from .nodes import ORDER
from .rules import is_protected, top_n
from .steps import memory
from .store import Store, parse_ts

WEB = Path(__file__).resolve().parent / "web"
DEMO_DIR = config.STATE / "demo-data"
NODE_ROLE = {"memory": "classify", "trends": "trends", "scouts": "scout", "dedupe": "dedupe", "critic1": "critic1",
             "reviser": "reviser", "critic2": "critic2", "logistics": "logistics", "demand": "demand",
             "judge": "judge", "factcheck": "factcheck", "close": None}


def data_dir(ds: str) -> Path | None:
    if ds == "demo":
        ensure_demo()
        return DEMO_DIR
    return config.DATA if (config.DATA / "ideas").is_dir() else None


def ensure_demo(reset: bool = False) -> None:
    if reset and DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    if not DEMO_DIR.exists():
        shutil.copytree(config.SAMPLE_DATA, DEMO_DIR, ignore=shutil.ignore_patterns("cache", "evals", "inbox"))


# ---- run process --------------------------------------------------------------------------------
class Runner:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.ds = ""
        self.cmd: list[str] = []
        self.log_path = config.STATE / "serve-run.log"
        self.lock = threading.Lock()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, ds: str, args: list[str], env: dict) -> None:
        with self.lock:
            if self.running():
                raise RuntimeError("A run is already in progress.")
            config.STATE.mkdir(parents=True, exist_ok=True)
            self.cmd = [sys.executable, "-m", "idea_lab", "run", *args]
            self.ds = ds
            log = self.log_path.open("w")
            self.proc = subprocess.Popen(self.cmd, cwd=config.ROOT, env={**os.environ, **env, "PYTHONUNBUFFERED": "1"},
                                         stdout=log, stderr=subprocess.STDOUT)

    def stop(self) -> None:
        with self.lock:
            if self.running():
                self.proc.send_signal(signal.SIGINT)

    def status(self) -> dict:
        tail = ""
        if self.log_path.exists():
            tail = "\n".join(self.log_path.read_text(errors="replace").splitlines()[-25:])
        return {"running": self.running(), "dataset": self.ds, "cmd": " ".join(self.cmd[2:]),
                "exit_code": None if self.proc is None or self.running() else self.proc.returncode, "tail": tail}


RUNNER = Runner()


def plan_args(ds: str, plan: str, store: Store) -> list[str]:
    args = ["--backend", "demo"] if ds == "demo" else []
    if ds == "demo":
        args += ["--force", "--run-id", datetime.now(timezone.utc).strftime("demo-%Y%m%d-%H%M%S")]
    if plan == "maintenance":
        args += ["--type", "maintenance"]
    elif plan == "judge":
        args += ["--only", "judge,factcheck", "--judge-scope", "validated"]
        runs = store.runs()
        if runs and runs[-1].get("status") == "stopped":
            args += ["--run-id", runs[-1]["_id"]]
    return args


def env_for(ds: str, speed: float | None = None) -> dict:
    env = {"IDEA_LAB_DATA": str(data_dir(ds))}
    if ds == "demo":
        env["IDEA_LAB_DEMO_DELAY"] = str(speed if speed is not None else 1.0)
    return env


# ---- state for the page -------------------------------------------------------------------------
def state(ds: str) -> dict:
    root = data_dir(ds)
    if root is None:
        return {"dataset": ds, "available": False, "runner": RUNNER.status()}
    store = Store(root)
    ideas = store.ideas()
    runs = store.runs()
    run = runs[-1] if runs else None
    started = parse_ts(run.get("started_at")) if run else None
    rows = []
    for i, d in ideas.items():
        h = (d.get("history") or [{}])[-1]
        touched = bool(started and any((t := parse_ts(x.get("at"))) and t >= started for x in d.get("history") or []))
        rows.append({"id": i, "title": d.get("title") or i, "stage": d.get("stage"), "total": d.get("total"),
                     "kind": d.get("kind"), "lane": d.get("lane"), "source": d.get("source"),
                     "updated_at": d.get("updated_at") or "", "rev_count": d.get("rev_count") or 0,
                     "last": {"by": h.get("by"), "note": (h.get("note") or "")[:140], "at": h.get("at")},
                     "touched": touched, "decided": is_protected(d)})
    q = memory.queues(ideas, datetime.now(timezone.utc))
    run_out = None
    if run:
        run_out = {k: run.get(k) for k in ("_id", "status", "run_type", "started_at", "ended_at", "phase", "plan",
                                           "nodes", "usage", "stopped_reason", "stopped_at_phase", "digest", "backend",
                                           "counts", "angles")}
        run_out["log"] = (run.get("log") or [])[-120:]
    return {
        "dataset": ds, "available": True, "data_dir": str(root), "ideas": rows, "top6": top_n(ideas, 6),
        "queues": {k: len(v) for k, v in q.items() if k != "top6"}, "run": run_out, "runs": [r["_id"] for r in runs[-10:]],
        "order": ORDER, "models": {n: (config.ROLE_MODEL.get(r) if r else None) for n, r in NODE_ROLE.items()},
        "budget": {"tokens": config.MAX_TOKENS_PER_RUN, "searches": config.MAX_SEARCHES_PER_RUN, "usd": config.MAX_USD_PER_RUN},
        "runner": RUNNER.status(),
    }


# ---- HTTP ---------------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "fishtank"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(u.query).items()}
        ds = qs.get("ds", "demo")
        if u.path in ("/", "/index.html"):
            return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            return self._json(state(ds))
        if u.path == "/api/idea":
            root = data_dir(ds)
            d = Store(root).get("ideas", qs.get("id", "")) if root else None
            return self._json(d or {"error": "not found"}, 200 if d else 404)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.headers.get("X-Fishtank") != "1":
            return self._json({"error": "missing X-Fishtank header"}, 403)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad JSON"}, 400)
        ds = body.get("ds", "demo")
        path = urlparse(self.path).path
        try:
            if path == "/api/run":
                root = data_dir(ds)
                if root is None:
                    return self._json({"error": "No real data folder (clone the data repo into data/)."}, 400)
                args = plan_args(ds, body.get("plan", "full"), Store(root))
                RUNNER.start(ds, args, env_for(ds, body.get("speed")))
                return self._json({"ok": True, "cmd": " ".join(args)})
            if path == "/api/dryrun":
                root = data_dir(ds)
                if root is None:
                    return self._json({"error": "No real data folder."}, 400)
                args = [a for a in plan_args(ds, body.get("plan", "full"), Store(root))] + ["--dry-run"]
                p = subprocess.run([sys.executable, "-m", "idea_lab", "run", *args], cwd=config.ROOT,
                                   env={**os.environ, **env_for(ds)}, capture_output=True, text=True, timeout=120)
                return self._json({"ok": p.returncode == 0, "output": (p.stdout + p.stderr).strip()})
            if path == "/api/stop":
                RUNNER.stop()
                return self._json({"ok": True})
            if path == "/api/reset-demo":
                if RUNNER.running():
                    return self._json({"error": "Stop the run first."}, 409)
                ensure_demo(reset=True)
                return self._json({"ok": True})
            if path == "/api/decide":
                if RUNNER.running():
                    return self._json({"error": "Wait for the run to finish before deciding."}, 409)
                root = data_dir(ds)
                msg = decisions.decide(Store(root), body["id"], body["status"], body.get("reasons") or [], body.get("text", ""))
                return self._json({"ok": True, "msg": msg})
        except (RuntimeError, KeyError, ValueError) as e:
            return self._json({"error": str(e)}, 400)
        self._json({"error": "not found"}, 404)


def _already_ours(url: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(url + "/api/state?ds=demo", timeout=2) as r:
            return "dataset" in json.loads(r.read())
    except Exception:  # noqa: BLE001
        return False


def serve(port: int = 8765, open_browser: bool = False) -> int:
    import errno
    import webbrowser

    ensure_demo()
    url = f"http://127.0.0.1:{port}"
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        if _already_ours(url):
            print(f"The Fishtank dashboard is already running at {url}", flush=True)
            if open_browser:
                webbrowser.open(url)
            return 0
        print(f"Port {port} is in use by something else. Try: python -m idea_lab serve --port {port + 1}", file=sys.stderr)
        return 2
    print(f"Fishtank dashboard on {url}  (Ctrl-C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        RUNNER.stop()
        httpd.server_close()
    return 0
