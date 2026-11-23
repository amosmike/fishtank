"""Command line: python -m idea_lab <command>. See README.md."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone

from . import config, llm
from .decisions import apply_all
from .graph import make_plan, run as run_graph
from .nodes import Ctx, judge_targets
from .rules import leaderboard, stale
from .steps import memory
from .store import Store, now_iso, parse_ts


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---- plan / dry run -------------------------------------------------------------------------------
def estimate(store: Store, plan: list[str], run_type: str, judge_scope: str) -> list[dict]:
    """Planned model calls per node, from today's queues. New-idea counts are upper bounds."""
    now = _now()
    ideas = store.ideas()
    q = memory.queues(ideas, now)
    im = (store.config("idea_map") or {}).get("ideas", {})
    discovery = run_type == "discovery" and not memory.saturated(ideas)
    new = config.MAX_NEW_IDEAS if ("dedupe" in plan and discovery) else 0
    rows = []

    def add(node, role, calls, items=""):
        if node in plan and calls > 0:
            rows.append({"node": node, "role": role, "calls": calls, "items": items})

    add("memory", "classify", math.ceil(sum(1 for i, d in ideas.items() if not d.get("category") and i not in im) / 80), "unclassified ideas")
    trends_stale = stale((store.config("trends") or {}).get("at"), now, config.TRENDS_MAX_AGE_DAYS)
    add("trends", "trends", int(discovery and trends_stale), "weekly cache stale" if trends_stale else "")
    add("scouts", "scout", config.SCOUTS if discovery else 0, f"{config.SCOUTS} scouts")
    add("dedupe", "dedupe", int(discovery), "")
    c1 = len(q["critic1"]) + len(q["critic1_fill"]) + new
    add("critic1", "critic1", math.ceil(c1 / config.CRITIC_BATCH), f"{c1} ideas ({len(q['critic1'])} queued, {len(q['critic1_fill'])} carry-over fills, up to {new} new)")
    rv = (len(q["reviser"]) + len(q["revive_parked"]) + round(new * 0.6)) if "reviser" in plan else 0
    add("reviser", "reviser", math.ceil(rv / 10), f"~{rv} ideas ({len(q['reviser'])} due now)")
    c2 = (len(q["critic2"]) + rv) if "critic2" in plan else 0
    add("critic2", "critic2", math.ceil(c2 / config.CRITIC_BATCH), f"~{c2} ideas")
    lg = min(config.LOGISTICS_TARGETS, len(q["logistics"]) + round(new * 0.5))
    add("logistics", "logistics", math.ceil(lg / config.LOGISTICS_BATCH), f"~{lg} ideas ({len(q['logistics'])} due now)")
    add("demand", "demand", int(bool(q["demand"]) or new > 0), f"{len(q['demand'])} targets now")
    jt = judge_targets(ideas, judge_scope)
    jn = len(jt) + round(new * 0.4) + round(c2 * 0.5)
    add("judge", "judge", math.ceil(jn / 8) if jn else 0, f"~{jn} ideas (scope {judge_scope}: {', '.join(jt[:8])}{'…' if len(jt) > 8 else ''})")
    add("factcheck", "factcheck", 1 if memory.factcheck_targets(ideas, now) or jn else 0, "top 3 needing a check")
    for r in rows:
        tin, tout = config.EST_TOKENS[r["role"]]
        model = config.ROLE_MODEL[r["role"]]
        pin, pout = config.PRICE_PER_MTOK.get(model, (0, 0))
        s = config.ROLE_SEARCHES.get(r["role"], 0) * r["calls"]
        r.update(model=model, tokens=(tin + tout) * r["calls"], searches_max=s,
                 usd=round(r["calls"] * (tin / 1e6 * pin + tout / 1e6 * pout) + s * config.PRICE_PER_SEARCH, 2))
    return rows


def print_estimate(rows: list[dict]) -> None:
    print(f"{'node':<10} {'model':<18} {'calls':>5} {'~tokens':>9} {'≤searches':>9} {'~$':>6}  what")
    for r in rows:
        print(f"{r['node']:<10} {r['model']:<18} {r['calls']:>5} {r['tokens']:>9,} {r['searches_max']:>9} {r['usd']:>6.2f}  {r['items']}")
    t = sum(r["tokens"] for r in rows)
    s = sum(r["searches_max"] for r in rows)
    usd = sum(r["usd"] for r in rows)
    print(f"{'TOTAL':<10} {'':<18} {sum(r['calls'] for r in rows):>5} {t:>9,} {s:>9} {usd:>6.2f}")
    print(f"Budget: {config.MAX_TOKENS_PER_RUN:,} tokens, {config.MAX_SEARCHES_PER_RUN} searches, "
          f"${config.MAX_USD_PER_RUN:.2f} (the run skips to judge -> close when any is hit). "
          f"Baseline: the 21–22 Nov run used ~2M tokens and ~25 agents.")
    if config.BACKEND == "cli":
        print("Backend cli: calls go through `claude -p` on your subscription; $ is the API-equivalent, not a bill.")


# ---- commands -------------------------------------------------------------------------------------
def cmd_run(a) -> int:
    store = Store()
    if a.backend:
        config.BACKEND = a.backend
    if a.scouts is not None:
        config.SCOUTS = a.scouts
    if a.max_tokens:
        config.MAX_TOKENS_PER_RUN = a.max_tokens
    if a.max_searches:
        config.MAX_SEARCHES_PER_RUN = a.max_searches
    if a.max_usd is not None:
        config.MAX_USD_PER_RUN = a.max_usd
    scope = "all" if a.calibrate else a.judge_scope
    only = [x.strip() for x in a.only.split(",")] if a.only else None

    if a.resume:
        run_id = a.resume
        run = store.run(run_id)
        if not run:
            print(f"No run {run_id}", file=sys.stderr)
            return 2
        budget = llm.Budget.from_dict(run.get("usage"), max_tokens=config.MAX_TOKENS_PER_RUN,
                                      max_searches=config.MAX_SEARCHES_PER_RUN)
        store.update_run(run_id, {"status": "running", "resumed_at": now_iso()},
                         log=[{"agent": "orchestrator", "msg": "Resumed"}])
        return _execute(store, run_id, budget, None, True, a.engine)

    plan = make_plan(a.type, only)
    if a.dry_run:
        print(f"Plan ({a.type}{', calibration' if a.calibrate else ''}): {' -> '.join(plan)}\n")
        print_estimate(estimate(store, plan, a.type, scope))
        return 0

    if not a.force:
        for r in store.runs():
            t = parse_ts(r.get("started_at"))
            if r.get("status") == "running" and t and _now() - t < timedelta(minutes=30) and r["_id"] != a.run_id:
                print(f"{r['_id']} started {r.get('started_at')} and is still running. Use --force to start anyway.",
                      file=sys.stderr)
                return 2

    run_id = a.run_id or _now().strftime("run-%Y%m%d-%H%M")
    existing = store.run(run_id)
    if existing:
        budget = llm.Budget.from_dict(existing.get("usage"), max_tokens=config.MAX_TOKENS_PER_RUN,
                                      max_searches=config.MAX_SEARCHES_PER_RUN)
        store.update_run(run_id, {"status": "running", "resumed_at": now_iso(), "playbook_version": config.PLAYBOOK_VERSION},
                         log=[{"agent": "orchestrator", "msg": f"Continued in code: {' -> '.join(plan)}"}])
    else:
        budget = llm.Budget()
        store.put("runs", run_id, {
            "started_at": now_iso(), "status": "running", "run_type": a.type, "playbook_version": config.PLAYBOOK_VERSION,
            "trigger": "manual", "backend": config.BACKEND, "calibration": bool(a.calibrate), "plan": plan,
            "counts": {}, "log": [], "agent_stats": {}, "angles": [],
        })
    initial = {"run_id": run_id, "run_type": a.type, "plan": plan, "judge_scope": scope, "done": []}
    return _execute(store, run_id, budget, initial, False, a.engine)


def _execute(store: Store, run_id: str, budget: llm.Budget, initial, resume: bool, engine: str) -> int:
    ctx = Ctx(store, budget, run_id)
    print(f"{run_id}: backend {config.BACKEND}, budget {budget.max_tokens:,} tokens / {budget.max_searches} searches")
    try:
        run_graph(ctx, initial, resume=resume, engine=engine)
    except (llm.LimitError, KeyboardInterrupt) as e:
        phase = store.run(run_id).get("phase")
        ctx.save_usage()
        store.update_run(run_id, {"status": "stopped", "stopped_at_phase": phase, "stopped_at": now_iso()},
                         log=[{"agent": "orchestrator", "msg": f"Stopped at {phase}: {e or 'interrupted'}"}])
        print(f"\nStopped at {phase}: {e or 'interrupted'}\nResume with: python -m idea_lab run --resume {run_id}",
              file=sys.stderr)
        return 3
    except Exception as e:  # a bug, not a limit: record it so the run isn't left looking "running"
        phase = store.run(run_id).get("phase")
        ctx.save_usage()
        store.update_run(run_id, {"status": "failed", "stopped_at_phase": phase, "stopped_at": now_iso()},
                         log=[{"agent": "orchestrator", "msg": f"Failed at {phase}: {type(e).__name__}: {e}"}])
        raise
    u = budget.total
    print(f"\nUsage: {u.tokens:,} tokens, {u.searches} searches, {u.calls} calls, ~${u.cost_usd:.2f}")
    return 0


def cmd_status(a) -> int:
    store = Store()
    ideas = store.ideas()
    print("Leaderboard:")
    for n, (i, d) in enumerate(leaderboard(ideas)[: a.n], 1):
        print(f"  {n:>2}. {d.get('total'):>3}  {d.get('title') or i}  [{i}, {d.get('stage')}]")
    q = memory.queues(ideas, _now())
    print("\nQueues: " + ", ".join(f"{k} {len(v)}" for k, v in q.items() if k != "top6"))
    for k in ("critic1", "reviser", "critic2", "validated"):
        if q[k]:
            print(f"  {k}: {', '.join(q[k])}")
    runs = store.runs()
    if runs:
        r = runs[-1]
        u = (r.get("usage") or {}).get("total") or {}
        stopped = r.get("status") in ("stopped", "failed") and r.get("stopped_at_phase")
        print(f"\nLast run {r['_id']}: {r.get('status')}" + (f" at {r.get('stopped_at_phase')}" if stopped else "")
              + (f", {u.get('tokens', 0):,} tokens, {u.get('searches', 0)} searches" if u else ""))
    return 0


def cmd_apply(a) -> int:
    for line in apply_all(Store()) or ["Nothing to apply."]:
        print(line)
    return 0


def cmd_eval(a) -> int:
    from . import evals
    store = Store()
    if a.suite == "retrieval":  # local, no model calls: BM25 vs vector vs hybrid on labels from the data
        from .retrieval import evaluate
        report = evaluate.run(store)
        print(evaluate.table(report))
        if not a.dry_run:
            print(f"\nSaved {evaluate.save(store, report)}")
        return 0
    if a.list:
        for r in evals.history(store) or []:
            print(evals.summary_line(r))
        return 0
    if a.model:
        config.ROLE_MODEL["judge"] = a.model
    p = evals.plan(store, a.repeats, a.folds)
    print(f"Judge eval: {p['ideas']} decided ideas, {a.folds} folds x {a.repeats} repeats = {p['calls']} calls "
          f"to {p['model']} (~{p['est_tokens']:,} tokens, ~${p['est_usd']:.2f} API-equivalent).")
    if a.dry_run:
        return 0
    report = evals.run_judge_eval(store, a.repeats, a.folds, budget=llm.Budget(max_usd=a.max_usd))
    path = evals.save(store, report)
    print(json.dumps(report["metrics"], indent=1))
    print(f"\n{evals.summary_line(report)}\nSaved {path}")
    return 0


def cmd_serve(a) -> int:
    from .serve import serve
    return serve(a.port, a.open)


def cmd_view(a) -> int:
    from .view import build
    path = build(Store(), config.OUT)
    print(f"Wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m idea_lab", description="Idea Lab pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline (or --dry-run to see the plan and cost)")
    r.add_argument("--type", choices=["discovery", "maintenance"], default="discovery")
    r.add_argument("--only", help="comma-separated nodes, e.g. judge,factcheck (memory and close always run)")
    r.add_argument("--run-id", help="continue an existing run doc (e.g. the paused run-20261121-0953)")
    r.add_argument("--resume", metavar="RUN_ID", help="resume a stopped run from its last checkpoint")
    r.add_argument("--dry-run", action="store_true", help="print planned calls and an estimated cost; no calls")
    r.add_argument("--scouts", type=int, help=f"number of scouts (default {config.SCOUTS})")
    r.add_argument("--max-tokens", type=int, help="run token budget")
    r.add_argument("--max-searches", type=int, help="run search budget")
    r.add_argument("--max-usd", type=float, help="run cost budget in USD (0 = no cost cap)")
    r.add_argument("--judge-scope", choices=["validated", "changed", "all"], default="changed",
                   help="validated: only newly validated ideas; changed: plus carry-overs with new evidence (default); all: everything")
    r.add_argument("--calibrate", action="store_true", help="re-score every carry-over (judge scope all)")
    r.add_argument("--engine", choices=["auto", "langgraph", "plain"], default="auto")
    r.add_argument("--backend", choices=["cli", "demo"], help="model backend (default from IDEA_LAB_BACKEND or cli; demo = simulated agents)")
    r.add_argument("--force", action="store_true", help="start even if another run started in the last 30 minutes")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("status", help="leaderboard, queues and the last run")
    s.add_argument("-n", type=int, default=10)
    s.set_defaults(fn=cmd_status)
    sub.add_parser("apply", help="apply decisions.yaml and viewer change files now").set_defaults(fn=cmd_apply)
    sub.add_parser("view", help="build out/index.html").set_defaults(fn=cmd_view)

    sv = sub.add_parser("serve", help="local dashboard: watch runs move ideas through the pipeline")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--open", action="store_true", help="open the browser")
    sv.set_defaults(fn=cmd_serve)

    e = sub.add_parser("eval", help="evals: the judge against the owner's decisions, or retrieval quality")
    e.add_argument("--suite", choices=["judge", "retrieval"], default="judge",
                   help="judge: taste agreement + stability (model calls); retrieval: BM25/vector/hybrid recall (local, free)")
    e.add_argument("--repeats", type=int, default=3, help="identical runs per idea, for stability (default 3)")
    e.add_argument("--folds", type=int, default=3, help="folds for leave-fold-out taste summaries (default 3)")
    e.add_argument("--model", help=f"judge model to evaluate (default {config.ROLE_MODEL['judge']})")
    e.add_argument("--max-usd", type=float, default=5.0, help="stop the eval past this API-equivalent cost")
    e.add_argument("--dry-run", action="store_true", help="show the planned calls and cost only")
    e.add_argument("--list", action="store_true", help="list past eval reports")
    e.set_defaults(fn=cmd_eval)

    a = p.parse_args(argv)
    if a.cmd != "serve" and not (config.DATA / "ideas").is_dir():
        print(f"No data at {config.DATA}. Clone the data repo into data/, or try the sample data:\n"
              f"  IDEA_LAB_DATA=sample_data python -m idea_lab {a.cmd}", file=sys.stderr)
        return 2
    return a.fn(a)
