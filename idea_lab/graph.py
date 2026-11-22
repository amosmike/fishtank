"""Wiring: the run as a DAG with parallel branches, and two engines to execute it.

    memory ─┬─ trends ── scouts ── dedupe ──────────────────┐
            └─ critic1·queued ── reviser·queued ── critic2·queued ┴─ critic1 ── reviser ── critic2 ─┬─ logistics ─┬─ judge ── factcheck ── close
                                                                                                    └─ demand ────┘

- The queued branch works on what was already waiting when the run started (the owner's submitted
  ideas, due revisions, revised-but-unchecked ideas) while trends and scouts research. It doesn't
  depend on them.
- Within one idea the order is fixed: critic 1 -> reviser -> critic 2 (critic 2 checks the reviser's
  answer to critic 1). So critic 1 and critic 2 run side by side only across different ideas.
- Scouts run in parallel with each other inside their node, and meet at dedupe: dedupe has to see the
  whole batch to catch two scouts proposing the same idea and to apply the per-run caps.
- Logistics and demand are independent checks on the same ideas (different fields), so they run in
  parallel and join at the judge, which needs everything to score on one scale.

Parallelism changes the wall-clock time, not the cost: every model call in the process shares
config.MAX_PARALLEL slots (llm._SLOTS), and every idea write takes a per-file lock (store.doc_lock).

Plan and budget are applied inside each node rather than by routing, so the DAG is static: a node
whose step isn't in the run's plan is marked "off", and once a budget is hit every model step before
the judge marks itself "skipped" (the old timebox rule: always finish with a ranking).

Engines:
- "langgraph" (default when installed): StateGraph with a SQLite checkpointer, thread_id = run id.
  Parallel branches run concurrently; a run stopped by a usage limit resumes from the last finished nodes.
- "plain": the same nodes one at a time in topological order; resume state is kept in the run file.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from . import config, llm
from .nodes import NODES as STEPS
from .nodes import ORDER, Ctx

# graph node -> (step function, predecessors). Step names are what a plan lists (see ORDER).
DAG: dict[str, tuple[str, list[str]]] = {
    "memory": ("memory", []),
    "trends": ("trends", ["memory"]),
    "scouts": ("scouts", ["trends"]),
    "dedupe": ("dedupe", ["scouts"]),
    "critic1_queued": ("critic1", ["memory"]),
    "reviser_queued": ("reviser", ["critic1_queued"]),
    "critic2_queued": ("critic2", ["reviser_queued"]),
    "critic1": ("critic1", ["dedupe", "critic2_queued"]),
    "reviser": ("reviser", ["critic1"]),
    "critic2": ("critic2", ["reviser"]),
    "logistics": ("logistics", ["critic2"]),
    "demand": ("demand", ["critic2"]),
    "judge": ("judge", ["logistics", "demand"]),
    "factcheck": ("factcheck", ["judge"]),
    "close": ("close", ["factcheck"]),
}
TOPO = list(DAG)  # insertion order above is a valid topological order
# Columns for drawing the graph: nodes in the same column can run at the same time.
LAYOUT = [["memory"], ["trends", "critic1_queued"], ["scouts", "reviser_queued"], ["dedupe", "critic2_queued"],
          ["critic1"], ["reviser"], ["critic2"], ["logistics", "demand"], ["judge"], ["factcheck"], ["close"]]

ALWAYS = ("memory", "close")


def _check_dag() -> None:
    seen: set[str] = set()
    for n in TOPO:
        step, preds = DAG[n]
        assert step in STEPS, n
        assert all(p in seen for p in preds), f"{n} listed before its predecessors"
        seen.add(n)
    assert sorted(n for col in LAYOUT for n in col) == sorted(TOPO)


_check_dag()


def _node(name: str):
    step, _ = DAG[name]
    fn = STEPS[step]

    def run(state: dict, ctx: Ctx) -> dict:
        if step not in (state.get("plan") or ORDER):
            ctx.node_status(name, "off")
            return {"done": [name]}
        token = llm.NODE.set(name)
        ctx.patch_run({"phase": name})
        ctx.node_status(name, "running")
        try:
            out = fn(state, ctx)
        except llm.ModelError as e:  # one bad response can't sink a run
            ctx.log("error", f"{name}: {e}")
            ctx.save_usage()
            ctx.node_status(name, "error")
            return {"done": [name]}
        except BaseException:  # limits and Ctrl-C stop the run; it can be resumed
            ctx.node_status(name, "stopped")
            raise
        finally:
            llm.NODE.reset(token)
        ctx.save_usage()
        ctx.node_status(name, "done")
        return {**out, "done": [name]}

    return run


NODES = {name: _node(name) for name in TOPO}


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
        bad = [n for n in only if n not in ORDER]
        if bad:
            raise ValueError(f"unknown step(s) {bad}; choose from {ORDER}")
        return [n for n in ORDER if n in only or n in ALWAYS]
    if run_type == "maintenance":
        return [n for n in ORDER if n not in ("trends", "scouts", "dedupe")]
    return list(ORDER)


# ---- LangGraph engine ---------------------------------------------------------------------------
def build_langgraph(ctx: Ctx, checkpointer=None):
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(RunState)
    for name, fn in NODES.items():
        g.add_node(name, (lambda f: (lambda s: f(s, ctx)))(fn))
    g.add_edge(START, "memory")
    for name, (_, preds) in DAG.items():
        if len(preds) == 1:
            g.add_edge(preds[0], name)
        elif preds:
            g.add_edge(preds, name)  # join: waits for every predecessor
    g.add_edge("close", END)
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
        if eng.get("next") not in DAG:
            raise SystemExit(f"No unfinished plain-engine state for {ctx.run_id}; nothing to resume.")
        state, start = eng["state"], TOPO.index(eng["next"])
        print(f"Resuming {ctx.run_id} at {eng['next']}")
    else:
        state, start = dict(initial or {}), 0
    for name in TOPO[start:]:
        ctx.patch_run({"_engine": {"next": name, "state": state}})
        upd = NODES[name](state, ctx)
        for k, v in upd.items():
            state[k] = state.get(k, []) + v if k == "done" else v
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
