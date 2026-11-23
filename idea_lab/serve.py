"""Local dashboard: `python -m idea_lab serve`, then open http://127.0.0.1:8765

Watch a run move ideas through the pipeline: node status and usage, ideas changing stage on a board,
the live log, and each idea's full history. Two datasets:

- demo: a scratch copy of sample_data/ (in .state/demo-data) run with simulated agents. No network,
  no tokens. "Reset demo" restores the copy.
- real: your data/ folder, run through the real backend after a dry-run estimate and a confirm.

Runs are subprocesses of `python -m idea_lab run`, so Stop is the same as Ctrl-C (the run is marked
stopped and can be resumed). The server binds to 127.0.0.1 only and answers only requests addressed
to 127.0.0.1 or localhost (so a DNS-rebinding page can't reach it), and every POST needs an
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
from .graph import DAG, LAYOUT, release
from .rules import is_protected, top_n
from .steps import memory
from .store import Store, parse_ts

WEB = Path(__file__).resolve().parent / "web"
# Bump when the page needs something new from the server. The page shows a "restart the dashboard"
# banner when the server it talks to reports an older version.
API_VERSION = 3
PAGE: bytes = b""  # the page is read once at start-up, so it always matches this server's code
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
        self.mode = ""
        self.cmd: list[str] = []
        self.log_path = config.STATE / "serve-run.log"
        self.lock = threading.Lock()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, ds: str, args: list[str], env: dict, mode: str = "continuous") -> None:
        with self.lock:
            if self.running():
                raise RuntimeError("A run is already in progress.")
            config.STATE.mkdir(parents=True, exist_ok=True)
            self.cmd = [sys.executable, "-m", "idea_lab", "run", *args]
            self.ds, self.mode = ds, mode
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
        return {"running": self.running(), "dataset": self.ds, "mode": self.mode, "cmd": " ".join(self.cmd[2:]),
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


def env_for(ds: str, mode: str = "continuous") -> dict:
    env = {"IDEA_LAB_DATA": str(data_dir(ds))}
    if ds == "demo":
        env["IDEA_LAB_DEMO_DELAY"] = "0.8"  # long enough to see each step working
    if mode == "step":
        env["IDEA_LAB_STEP"] = "1"
    return env


# ---- state for the page -------------------------------------------------------------------------
def _killed_at(d: dict) -> dict:
    h = next((h for h in reversed(d.get("history") or []) if h.get("stage") == "killed"), {})
    return {"at": h.get("at") or d.get("updated_at"), "by": h.get("by") or ""}


# Which history authors belong to a step, for "ideas this task touched".
STEP_AUTHORS = {"memory": ("Michael", "orchestrator"), "dedupe": ("scout",), "critic1": ("critic1",), "reviser": ("reviser", "orchestrator"),
                "critic2": ("critic2",), "logistics": ("logistics",), "demand": ("demand",), "judge": ("judge",),
                "factcheck": ("factcheck",)}
STEP_AGENTS = {"memory": ("memory", "decisions")}


def node_detail(store: Store, nodes: list[str]) -> dict:
    """What each graph node did in the latest run: timing, usage, log lines, ideas touched, extras."""
    runs = store.runs()
    run = runs[-1] if runs else {}
    rn, by = run.get("nodes") or {}, (run.get("usage") or {}).get("by_node") or {}
    trace = (run.get("usage") or {}).get("trace") or []
    ideas = store.ideas()
    out = []
    for n in nodes:
        if n not in DAG:
            continue
        step = DAG[n][0]
        info = rn.get(n) or {}
        t0, t1 = info.get("started_at") or "", info.get("ended_at") or ""
        within = (lambda t: bool(t0) and t0 <= (t or "") and (not t1 or (t or "") <= t1 + "~"))
        agents = STEP_AGENTS.get(step, (step,))
        log = [e for e in run.get("log") or [] if e.get("agent") in agents + ("error",) and within(e.get("t"))
               and (e.get("agent") != "error" or step in (e.get("msg") or "") or n in (e.get("msg") or ""))]
        touched = []
        authors = STEP_AUTHORS.get(step, ())
        if authors and t0:
            for i, d in ideas.items():
                hs = [h for h in d.get("history") or [] if within(h.get("at")) and str(h.get("by") or "").startswith(authors)]
                if hs:
                    touched.append({"id": i, "title": d.get("title") or i, "stage_now": d.get("stage"),
                                    "notes": [f"{h.get('stage')}: {h.get('note') or ''}".strip(": ") for h in hs][-3:]})
        extra = {}
        if step == "scouts":
            extra["scouts"] = [{"scout": k, "angle": v.get("angle"), "ideas": [x.get("title") for x in v.get("ideas") or []],
                                "signals": len(v.get("signals") or [])} for k, v in (run.get("scout_output") or {}).items()]
        if step == "trends":
            extra["trends"] = run.get("trends") or []
        out.append({"node": n, "step": step, "status": info.get("status") or "", "started_at": t0, "ended_at": t1,
                    "model": config.ROLE_MODEL.get(NODE_ROLE[step]) if NODE_ROLE.get(step) else None,
                    "usage": by.get(n) or {}, "log": log[-12:], "ideas": sorted(touched, key=lambda x: x["title"].lower()),
                    "preds": DAG[n][1], "calls": [c for c in trace if c.get("node") == n],
                    **extra})
    return {"run": run.get("_id"), "nodes": out}

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
                     "touched": touched, "decided": is_protected(d),
                     **({"stage_note": (d.get("stage_note") or "")[:240], "killed_at": _killed_at(d)} if d.get("stage") == "killed" else {})})
    q = memory.queues(ideas, datetime.now(timezone.utc))
    run_out = None
    if run:
        run_out = {k: run.get(k) for k in ("_id", "status", "run_type", "started_at", "ended_at", "phase", "plan",
                                           "nodes", "usage", "stopped_reason", "stopped_at_phase", "digest", "backend", "summary",
                                           "counts", "angles")}
        run_out["log"] = (run.get("log") or [])[-120:]
    return {
        "dataset": ds, "available": True, "data_dir": str(root), "ideas": rows, "top6": top_n(ideas, 6),
        "queues": {k: len(v) for k, v in q.items() if k != "top6"}, "run": run_out, "runs": [r["_id"] for r in runs[-10:]],
        "layout": LAYOUT, "steps": {n: s for n, (s, _) in DAG.items()}, "dag": {n: p for n, (_, p) in DAG.items()},
        "models": {n: (config.ROLE_MODEL.get(NODE_ROLE[s]) if NODE_ROLE[s] else None) for n, (s, _) in DAG.items()},
        "max_parallel": config.MAX_PARALLEL, "api": API_VERSION,
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

    def _local(self) -> bool:
        port = self.server.server_address[1]
        if self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return True
        self._json({"error": "bad Host header"}, 403)
        return False

    def do_GET(self):
        if not self._local():
            return
        u = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(u.query).items()}
        ds = qs.get("ds", "demo")
        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE or (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            return self._json(state(ds))
        if u.path == "/api/nodes":
            root = data_dir(ds)
            if root is None:
                return self._json({"error": "no data"}, 404)
            return self._json(node_detail(Store(root), [x for x in qs.get("nodes", "").split(",") if x]))
        if u.path == "/api/idea":
            root = data_dir(ds)
            d = Store(root).get("ideas", qs.get("id", "")) if root else None
            return self._json(d or {"error": "not found"}, 200 if d else 404)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if not self._local():
            return
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
                mode = "step" if body.get("mode") == "step" else "continuous"
                args = plan_args(ds, body.get("plan", "full"), Store(root))
                RUNNER.start(ds, args, env_for(ds, mode), mode)
                return self._json({"ok": True, "cmd": " ".join(args)})
            if path == "/api/next":
                root = data_dir(ds)
                runs = Store(root).runs() if root else []
                if not runs or not RUNNER.running():
                    return self._json({"error": "No run in progress."}, 409)
                run = runs[-1]
                waiting = [n for n, v in (run.get("nodes") or {}).items() if v.get("status") == "waiting"]
                if waiting:
                    release(run["_id"], waiting)
                return self._json({"ok": True, "released": waiting})
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

    global PAGE
    PAGE = (WEB / "index.html").read_bytes()
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
