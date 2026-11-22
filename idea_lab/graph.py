"""Wiring: the node order, budget routing, and two engines.

- "langgraph" (default when installed): a StateGraph with a SQLite checkpointer. thread_id = run id,
  so a run killed by a rate or usage limit resumes from the last finished node.
- "plain": the same nodes in a loop; resume state is saved in the run file under `_engine`.

Routing rule (the old timebox rule): when the token or search budget is hit, skip every remaining
node except judge and close.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from . import config, llm
from .nodes import NODES as _NODES
from .nodes import ORDER, Ctx


def _safe(name: str, fn):
    """A model error inside a node is logged and the node is marked done, so one bad response
    can't sink a run. Limit errors still propagate: they stop the run so it can be resumed."""
    def run(state: dict, ctx: Ctx) -> dict:
        ctx.node_status(name, "running")
        try:
            out = fn(state, ctx)
        except llm.ModelError as e:
            ctx.log("error", f"{name}: {e}")
            ctx.save_usage()
            ctx.node_status(name, "error")
            return {"done": [name]}
        except BaseException:
            ctx.node_status(name, "stopped")
            raise
        ctx.save_usage()
        ctx.node_status(name, "done")
        return out
    return run


NODES = {name: _safe(name, fn) for name, fn in _NODES.items()}

ALWAYS = ("memory", "close")
BUDGET_EXEMPT = ("judge", "close")


class RunState(TypedDict, total=False):
    run_id: str
    run_type: str
    plan: list[str]
    judge_scope: str
    trends_refreshed: bool
    top6_before: list[str]
    done: Annotated[list[str], operator.add]


def make_plan(run_type: str, only: list[str] | None = None) -> list[str]:
    if only:
        bad = [n for n in only if n not in NODES]
        if bad:
            raise ValueError(f"unknown node(s) {bad}; choose from {ORDER}")
        return [n for n in ORDER if n in only or n in ALWAYS]
    if run_type == "maintenance":
        return [n for n in ORDER if n not in ("trends", "scouts", "dedupe")]
    return list(ORDER)


def next_node(state: dict, current: str, ctx: Ctx) -> str | None:
    plan = state.get("plan") or ORDER
    over = ctx.budget.exceeded()
    for n in ORDER[ORDER.index(current) + 1:]:
        if n not in plan:
            continue
        if over and n not in BUDGET_EXEMPT:
            continue
        return n
    return None


# ---- LangGraph engine ---------------------------------------------------------------------------
def build_langgraph(ctx: Ctx, checkpointer=None):
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(RunState)
    for name, fn in NODES.items():
        g.add_node(name, (lambda f: (lambda s: f(s, ctx)))(fn))
    g.add_edge(START, "memory")
    for name in NODES:
        targets = {n: n for n in ORDER[ORDER.index(name) + 1:]}
        targets["__end__"] = END
        g.add_conditional_edges(name, (lambda cur: (lambda s: next_node(s, cur, ctx) or "__end__"))(name), targets)
    return g.compile(checkpointer=checkpointer)


def sqlite_checkpointer():
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    config.STATE.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.STATE / "checkpoints.sqlite", check_same_thread=False)
    return SqliteSaver(conn)


def run_langgraph(ctx: Ctx, initial: dict | None, resume: bool) -> dict:
    graph = build_langgraph(ctx, sqlite_checkpointer())
    cfg = {"configurable": {"thread_id": ctx.run_id}, "recursion_limit": 50}
    if resume:
        snap = graph.get_state(cfg)
        if not snap.next:
            raise SystemExit(f"No unfinished checkpoint for {ctx.run_id}; nothing to resume.")
        print(f"Resuming {ctx.run_id} at {', '.join(snap.next)}")
        return graph.invoke(None, cfg)
    return graph.invoke(initial, cfg)


# ---- plain engine -------------------------------------------------------------------------------
def run_plain(ctx: Ctx, initial: dict | None, resume: bool) -> dict:
    run = ctx.store.run(ctx.run_id)
    if resume:
        eng = run.get("_engine") or {}
        if not eng.get("next"):
            raise SystemExit(f"No unfinished plain-engine state for {ctx.run_id}; nothing to resume.")
        state, cur = eng["state"], eng["next"]
        print(f"Resuming {ctx.run_id} at {cur}")
    else:
        state, cur = dict(initial or {}), "memory"
    while cur:
        ctx.patch_run({"_engine": {"next": cur, "state": state}})
        upd = NODES[cur](state, ctx)
        for k, v in upd.items():
            state[k] = state.get(k, []) + v if k == "done" else v
        cur = next_node(state, cur, ctx)
    ctx.patch_run({"_engine": {"next": None, "state": state}})
    return state


def run(ctx: Ctx, initial: dict | None, resume: bool = False, engine: str = "auto") -> dict:
    if engine == "auto":
        try:
            import langgraph  # noqa: F401
            import langgraph.checkpoint.sqlite  # noqa: F401
            engine = "langgraph"
        except ImportError:
            engine = "plain"
    ctx.patch_run({"engine": engine})
    return run_langgraph(ctx, initial, resume) if engine == "langgraph" else run_plain(ctx, initial, resume)
