"""One way to call a model: `call(role, task, payload, schema, budget=...)` -> dict.

Two backends, picked by config.BACKEND / IDEA_LAB_BACKEND:
- "cli": headless Claude Code (`claude -p`) on Michael's subscription (no API key), locally and in CI
  (a `claude setup-token` token). Web search is Claude Code's WebSearch/WebFetch tools; searches are
  counted from the stream and capped by the prompt plus --max-turns.
- "demo": simulated agents (idea_lab/demo.py), no network, no tokens; for watching a run.

Every call records tokens and searches into the run's Budget, per node.
"""
from __future__ import annotations

import contextvars
import json
import os
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config


# The graph node currently running (set by graph._safe). Usage is recorded against it, so the two
# critic-1 nodes (queued ideas and new ideas) report their own numbers.
NODE: contextvars.ContextVar[str | None] = contextvars.ContextVar("idea_lab_node", default=None)

# Every model call in the process shares this many slots, however many graph branches and batches
# are running. Parallel branches make runs shorter; this keeps them from tripping rate limits.
_SLOTS = threading.BoundedSemaphore(config.MAX_PARALLEL)


class LimitError(Exception):
    """Usage/rate limit or login problem: stop the run (it can be resumed later)."""


class ModelError(Exception):
    """A single call failed (bad output, refusal, timeout)."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    searches: int = 0
    calls: int = 0
    cost_usd: float = 0.0  # Claude Code's API-equivalent cost (not a bill on a subscription)

    @property
    def tokens(self) -> int:
        """Everything processed, cache reads included (for display)."""
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def budget_tokens(self) -> int:
        """What the budget counts: fresh tokens plus a tenth of cache reads (they cost about 10%).
        Claude Code re-reads its cached context on every tool turn, so raw totals overstate use."""
        return self.input_tokens + self.output_tokens + self.cache_write_tokens + self.cache_read_tokens // 10

    def add(self, other: "Usage") -> None:
        for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "searches", "calls"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.cost_usd = round(self.cost_usd + other.cost_usd, 4)


@dataclass
class Budget:
    max_tokens: int = field(default_factory=lambda: config.MAX_TOKENS_PER_RUN)
    max_searches: int = field(default_factory=lambda: config.MAX_SEARCHES_PER_RUN)
    max_usd: float = field(default_factory=lambda: config.MAX_USD_PER_RUN)
    total: Usage = field(default_factory=Usage)
    by_node: dict[str, Usage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, node: str, u: Usage) -> None:
        with self._lock:
            self.total.add(u)
            self.by_node.setdefault(node, Usage()).add(u)

    def calls(self, node: str) -> int:
        with self._lock:
            return self.by_node[node].calls if node in self.by_node else 0

    def exceeded(self) -> str | None:
        if self.total.budget_tokens >= self.max_tokens:
            return f"token budget hit ({self.total.budget_tokens:,} >= {self.max_tokens:,})"
        if self.total.searches >= self.max_searches:
            return f"search budget hit ({self.total.searches} >= {self.max_searches})"
        if self.max_usd and self.total.cost_usd >= self.max_usd:
            return f"cost budget hit (${self.total.cost_usd:.2f} >= ${self.max_usd:.2f})"
        return None

    def to_dict(self) -> dict:
        def row(u: Usage) -> dict:
            return {**asdict(u), "tokens": u.tokens, "budget_tokens": u.budget_tokens}
        return {
            "max_tokens": self.max_tokens, "max_searches": self.max_searches, "max_usd": self.max_usd,
            "total": row(self.total), "by_node": {k: row(v) for k, v in self.by_node.items()},
        }

    @classmethod
    def from_dict(cls, d: dict | None, **overrides) -> "Budget":
        b = cls(**overrides)
        if not d:
            return b
        fields = set(Usage.__dataclass_fields__)
        b.total = Usage(**{k: v for k, v in (d.get("total") or {}).items() if k in fields})
        b.by_node = {k: Usage(**{kk: vv for kk, vv in v.items() if kk in fields}) for k, v in (d.get("by_node") or {}).items()}
        return b


# ---- prompts -----------------------------------------------------------------------------------
def load_prompt(role: str) -> str:
    common = (config.PROMPTS / "_rules.md").read_text()
    return (config.PROMPTS / f"{role}.md").read_text().strip() + "\n\n" + common.strip() + "\n"


def search_note(n: int) -> str:
    if n <= 0:
        return "You have NO web access for this task: work only from the material given."
    return (f"You may run at most {n} web searches for this whole task (fetching a page you found counts "
            f"as part of that search). Plan them; stop searching when you have enough.")


# ---- public entry point ------------------------------------------------------------------------
def call(role: str, payload: dict, schema: dict, *, node: str, budget: Budget,
         prompt_role: str | None = None, searches: int | None = None, extra: str = "",
         model: str | None = None) -> dict:
    model = model or config.ROLE_MODEL[role]
    effort = config.ROLE_EFFORT.get(role, "medium")
    n = config.ROLE_SEARCHES.get(role, 0) if searches is None else searches
    node = NODE.get() or node
    if config.BACKEND == "demo":
        from . import demo
        with _SLOTS:
            data, usage = demo.respond(role, payload, model, n)
        budget.record(node, usage)
        return data
    system = load_prompt(prompt_role or role)
    user = (f"{search_note(n)}\n{extra}\n\nReturn ONLY the JSON object described by the output schema.\n\n"
            f"TASK INPUT (JSON):\n{json.dumps(payload, ensure_ascii=False, indent=1)}")
    if config.BACKEND != "cli":
        raise LimitError(f"unknown backend {config.BACKEND!r} (use cli or demo)")
    with _SLOTS:
        data, usage = _cli_call(model, effort, system, user, schema, n)
    budget.record(node, usage)
    return data


# ---- CLI backend (claude -p) --------------------------------------------------------------------
_LIMIT_RE = ("usage limit", "rate limit", "limit reached", "429", "overloaded", "not logged in",
             "please run /login", "authenticat", "401")


def _cli_call(model: str, effort: str, system: str, user: str, schema: dict, searches: int):
    tools = "WebSearch,WebFetch" if searches > 0 else ""
    # The task goes in on stdin: as an argument it would hit Linux's 128 KB per-argument limit.
    argv = [config.CLAUDE_BIN, "-p", "--model", model, "--effort", effort,
            "--system-prompt", system, "--output-format", "stream-json", "--verbose",
            "--json-schema", json.dumps(schema), "--no-session-persistence",
            "--strict-mcp-config", "--disable-slash-commands", "--tools", tools]
    if tools:
        argv += ["--allowedTools", tools]
    # Each search plus its page fetches is a few turns; this is a ceiling, not a target.
    argv += ["--max-turns", str(max(4, searches * 3 + 4))]
    env = {**os.environ}
    env.pop("ANTHROPIC_API_KEY", None)   # always the subscription login
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    # A temp cwd so Claude Code doesn't pick up this repo's CLAUDE.md or settings.
    with tempfile.TemporaryDirectory(prefix="idea-lab-") as cwd:
        try:
            p = subprocess.run(argv, input=user, cwd=cwd, env=env, capture_output=True, text=True,
                               timeout=config.CALL_TIMEOUT_S)
        except FileNotFoundError as e:
            raise LimitError(f"can't run {config.CLAUDE_BIN!r}: is Claude Code installed?") from e
        except subprocess.TimeoutExpired as e:
            raise ModelError(f"claude timed out after {config.CALL_TIMEOUT_S}s") from e
    return parse_stream(p.stdout, p.stderr, p.returncode)


def parse_stream(stdout: str, stderr: str = "", code: int = 0):
    """Parse `claude -p --output-format stream-json` output -> (structured data, Usage)."""
    searches = 0
    result = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for block in (ev.get("message") or {}).get("content") or []:
                if block.get("type") == "tool_use" and block.get("name") == "WebSearch":
                    searches += 1
                if block.get("type") == "server_tool_use" and block.get("name") == "web_search":
                    searches += 1
        elif ev.get("type") == "result":
            result = ev
    if result is None:
        msg = (stderr or stdout).strip()[-400:]
        if any(s in msg.lower() for s in _LIMIT_RE):
            raise LimitError(msg)
        raise ModelError(f"claude exited {code} with no result: {msg}")
    u = result.get("usage") or {}
    stu = (u.get("server_tool_use") or {}).get("web_search_requests") or 0
    usage = Usage(
        input_tokens=int(u.get("input_tokens") or 0), output_tokens=int(u.get("output_tokens") or 0),
        cache_read_tokens=int(u.get("cache_read_input_tokens") or 0),
        cache_write_tokens=int(u.get("cache_creation_input_tokens") or 0),
        searches=max(searches, int(stu)), calls=1, cost_usd=float(result.get("total_cost_usd") or 0),
    )
    if result.get("is_error"):
        msg = str(result.get("result") or result.get("subtype") or "")
        if any(s in msg.lower() for s in _LIMIT_RE):
            raise LimitError(msg[:300])
        raise ModelError(msg[:300] or "claude reported an error")
    data = result.get("structured_output")
    if data is None:
        data = _json_from_text(str(result.get("result") or ""))
    if not isinstance(data, dict):
        raise ModelError("no structured output in claude result")
    return data, usage


def _json_from_text(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                return None
    return None
