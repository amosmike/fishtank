"""Graph nodes. Each is `node(state, ctx) -> state update` and works from queues computed from data,
so a node re-run after a crash only handles ideas it hasn't finished. Model calls go through llm.call.
"""
from __future__ import annotations

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from . import config, dice, llm, rules, schemas
from .checks import Cache, licence_fetch, rdap, url_check
from .decisions import apply_all as apply_decisions
from .rules import (apply_fact_check, apply_famous_caps, apply_stability, critic_outcome, has_new_evidence,
                    is_michaels, is_protected, leaderboard, logistics_outcome, stale, top_n, total)
from .steps import close as close_step
from .steps import dedupe as dedupe_step
from .steps import memory
from .store import Store, now_iso

ORDER = ["memory", "trends", "scouts", "dedupe", "critic1", "reviser", "critic2", "logistics", "demand",
         "judge", "factcheck", "close"]
MODEL_NODES = {"trends", "scouts", "dedupe", "critic1", "reviser", "critic2", "logistics", "demand", "judge", "factcheck"}

EVIDENCE_KEYS = ["title", "pitch", "kind", "lane", "effort_weeks", "running_cost", "search_phrase", "signals",
                 "competitors", "serp_note", "wedge", "gap_note", "demand_evidence", "audience_estimate",
                 "data_sources", "build_plan", "traffic_plan", "monetisation", "risks", "famous_original",
                 "why_big", "v1"]
CRITIC_FIELDS = ["competitors", "serp_note", "wedge", "gap_note", "demand_evidence", "audience_estimate",
                 "data_sources", "build_plan", "traffic_plan", "monetisation", "risks", "lane", "effort_weeks",
                 "running_cost", "famous_original"]


@dataclass
class Ctx:
    store: Store
    budget: llm.Budget
    run_id: str
    _lock: threading.Lock = field(default_factory=threading.Lock)
    errors: dict = field(default_factory=dict)  # graph node -> model errors logged while it ran

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def log(self, agent: str, msg: str) -> None:
        if agent == "error":
            node = llm.NODE.get() or "?"
            self.errors[node] = self.errors.get(node, 0) + 1
        with self._lock:
            self.store.update_run(self.run_id, log=[{"agent": agent, "msg": msg[:300]}])
        print(f"  [{agent}] {msg}", flush=True)

    def bump(self, counts: dict | None = None, role: str | None = None, stats: dict | None = None) -> None:
        with self._lock:
            r = self.store.run(self.run_id)
            c = r.setdefault("counts", {})
            for k, v in (counts or {}).items():
                c[k] = c.get(k, 0) + v
            if role:
                s = r.setdefault("agent_stats", {}).setdefault(role, {})
                for k, v in (stats or {}).items():
                    s[k] = s.get(k, 0) + v
            self.store.put("runs", self.run_id, r)

    def patch_run(self, patch: dict) -> None:
        with self._lock:
            self.store.update_run(self.run_id, patch)

    def save_usage(self) -> None:
        self.patch_run({"usage": self.budget.to_dict()})

    def node_status(self, name: str, status: str) -> None:
        """Record a node's status in the run file (running, done, skipped, error, stopped) for the dashboard.
        "skipped" set during a node sticks; "done" doesn't overwrite it."""
        with self._lock:
            r = self.store.run(self.run_id)
            nodes = r.setdefault("nodes", {})
            n = nodes.setdefault(name, {})
            if status in ("done", "idle") and n.get("status") == "skipped":
                return
            n["status"] = status
            n["started_at" if status == "running" else "ended_at"] = now_iso()
            self.store.put("runs", self.run_id, r)


# ---- helpers ------------------------------------------------------------------------------------
def chunks(xs: list, n: int) -> list[list]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def parallel(ctx: Ctx, node: str, fn: Callable, items: list) -> list:
    """Run fn over items with MAX_PARALLEL threads. ModelErrors are logged and skipped; a LimitError
    stops the run once the in-flight calls finish (their results are already saved)."""
    results, limit = [], None

    def wrap(x):
        try:
            return fn(x)
        except llm.ModelError as e:
            ctx.log("error", f"{node}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL) as ex:
        # each worker runs in a copy of this thread's context, so usage is recorded against this node
        futs = [ex.submit(contextvars.copy_context().run, wrap, x) for x in items]
        for f in futs:
            try:
                results.append(f.result())
            except llm.LimitError as e:
                limit = e
    ctx.save_usage()
    if limit:
        raise limit
    return results


def evidence(d: dict) -> dict:
    """The blind view of an idea: pitch and evidence only (no source, notes, revision count, history)."""
    e = {k: d.get(k) for k in EVIDENCE_KEYS if d.get(k) not in (None, "", [])}
    if d.get("demand_hunt"):
        dh = d["demand_hunt"]
        e["demand_hunt"] = {k: dh.get(k) for k in ("verdict", "suggested_demand_change", "note", "direct", "against")}
    if d.get("logistics"):
        lg = d["logistics"]
        e["logistics"] = {k: lg.get(k) for k in ("verdict", "effort_days", "upkeep_hours_week", "cost_monthly",
                                                 "free_tier_limit", "blockers", "fixes", "riskiest_part", "name_check",
                                                 "domain_check", "note")}
    if d.get("fact_check"):
        e["fact_check"] = {k: d["fact_check"].get(k) for k in ("status", "summary")}
    crit = d.get("critique2") or d.get("critique") or {}
    if isinstance(crit, str):
        e["critic_verdict"] = {"text": crit}
    elif crit:
        e["critic_verdict"] = {k: crit.get(k) for k in ("verdict", "fix_worked", "kill_reason")}
    return e


def critique(d: dict, key: str) -> dict:
    """A critic's verdict. Older ideas from the prose prototype store it as plain text."""
    v = d.get(key) or {}
    return {"text": v} if isinstance(v, str) else v


def trim(d: dict, keys: list[str]) -> dict:
    return {k: d.get(k) for k in keys if d.get(k) not in (None, "", [])}


def rerank(ctx: Ctx) -> list[str]:
    """Top 6 undecided scored ideas -> shortlisted, the rest -> judged."""
    ideas = ctx.store.ideas()
    board = leaderboard(ideas)
    top = [i for i, _ in board[:6]]
    for i, d in board:
        want = "shortlisted" if i in top else "judged"
        if d.get("stage") != want:
            ctx.store.update_idea(i, {"stage": want}, by="judge",
                                  note="Entered the top 6" if want == "shortlisted" else "Dropped out of the top 6")
    return top


def skip_if_over_budget(ctx: Ctx, node: str) -> bool:
    why = ctx.budget.exceeded()
    if why:
        ctx.log("orchestrator", f"Skipping {node}: {why}")
        ctx.patch_run({"stopped_reason": why})
        if llm.NODE.get():
            ctx.node_status(llm.NODE.get(), "skipped")
        return True
    return False


# ---- nodes --------------------------------------------------------------------------------------
def memory_node(state: dict, ctx: Ctx) -> dict:
    for line in apply_decisions(ctx.store):
        ctx.log("decisions", line)
    classify_missing(ctx)
    ideas = ctx.store.ideas()
    q = memory.queues(ideas, ctx.now())
    ctx.log("memory", "Queues: " + ", ".join(f"{k} {len(v)}" for k, v in q.items() if k != "top6") +
            f". Top 6: {', '.join(q['top6'])}.")
    return {"top6_before": q["top6"], "done": ["memory"]}


def classify_missing(ctx: Ctx) -> None:
    """Idea map: give every idea a category + mechanic (one cheap call for the ones that lack them)."""
    im = ctx.store.config("idea_map") or {"ideas": {}}
    ideas = ctx.store.ideas()
    todo = [i for i, d in ideas.items() if not d.get("category") and i not in im["ideas"]]
    if not todo:
        return
    known = sorted({v["category"] for v in im["ideas"].values()} | {d["category"] for d in ideas.values() if d.get("category")})
    for batch in chunks(todo, 80):
        payload = {"ideas": [{"id": i, "title": ideas[i].get("title"), "pitch": (ideas[i].get("pitch") or "")[:140],
                              "kind": ideas[i].get("kind")} for i in batch], "known_labels": known[:60]}
        out = llm.call("classify", payload, schemas.CLASSIFY, node="memory", budget=ctx.budget)
        for r in out.get("ideas") or []:
            if r.get("id") in ideas:
                im["ideas"][r["id"]] = {"category": r.get("category", ""), "mechanic": r.get("mechanic", "")}
    im["updated_at"] = now_iso()
    ctx.store.put("config", "idea_map", im)
    ctx.save_usage()
    ctx.log("memory", f"Idea map: classified {len(todo)} ideas")


def trends_node(state: dict, ctx: Ctx) -> dict:
    prev = ctx.store.config("trends") or {}
    if not stale(prev.get("at"), ctx.now(), config.TRENDS_MAX_AGE_DAYS):
        ctx.log("trends", f"Cache from {prev.get('at')} is fresh; reusing it")
        return {"done": ["trends"], "trends_refreshed": False}
    if skip_if_over_budget(ctx, "trends"):
        return {"done": ["trends"], "trends_refreshed": False}
    out = llm.call("trends", {"previous_trends": prev.get("trends") or [], "date": ctx.now().date().isoformat()},
                   schemas.TRENDS, node="trends", budget=ctx.budget)
    trends = out.get("trends") or []
    ctx.store.put("config", "trends", {"at": now_iso(), "run_id": ctx.run_id, "trends": trends})
    ctx.patch_run({"trends": [t.get("trend", "")[:120] for t in trends]})
    ctx.save_usage()
    ctx.log("trends", f"{len(trends)} trends refreshed")
    return {"done": ["trends"], "trends_refreshed": True}


def scouts_node(state: dict, ctx: Ctx) -> dict:
    ideas = ctx.store.ideas()
    if memory.saturated(ideas):
        ctx.log("orchestrator", "Saturation guard: 8+ ideas revising, skipping scouts (maintenance)")
        return {"done": ["scouts"]}
    if skip_if_over_budget(ctx, "scouts"):
        return {"done": ["scouts"]}
    run = ctx.store.run(ctx.run_id)
    briefs = run.get("scout_briefs")
    if not briefs:  # roll once per run, saved before any call, so a resume reuses the same rolls
        runs = [r for r in ctx.store.runs() if r["_id"] != ctx.run_id]
        n = max(1, config.SCOUTS)
        briefs = []
        if state.get("trends_refreshed") and n >= 2:
            briefs.append({"scout": "trends", "angle": "t: trends"})
        n_dice = max(0, n - len(briefs) - 1)
        for r in dice.roll(n_dice, dice.recent_pairs(runs)):
            briefs.append({"scout": r["scout"], "angle": f"dice: {r['audience']} / {r['format']} / {r['engine']}", "dice": r})
        letter = dice.pick_angle(runs)
        briefs.append({"scout": "angle", "angle": f"{letter}: {dice.ANGLES[letter]}"})
        stats = run.setdefault("agent_stats", {})
        stats["dice"] = [b["dice"] for b in briefs if b.get("dice")]
        ctx.patch_run({"scout_briefs": briefs, "angles": [b["angle"] for b in briefs], "agent_stats": stats})
        ctx.log("scouts", "Briefs: " + " | ".join(b["angle"] for b in briefs))
    done = set((ctx.store.run(ctx.run_id).get("scout_output") or {}).keys())
    todo = [b for b in briefs if b["scout"] not in done]
    mem = memory.brief(ctx.store, ctx.now(), "scout")
    trends = (ctx.store.config("trends") or {}).get("trends") or []

    def one(b):
        payload = {"brief": {k: v for k, v in b.items() if k != "scout"}, "memory": mem, "trends": trends}
        if b["scout"] == "trends":
            payload["brief"]["instruction"] = "Build ideas directly from the trends' idea seeds. A fading trend can't be the main evidence."
        out = llm.call("scout", payload, schemas.SCOUT, node="scouts", budget=ctx.budget)
        with ctx._lock:
            r = ctx.store.run(ctx.run_id)
            r.setdefault("scout_output", {})[b["scout"]] = {"angle": b["angle"], "dice": b.get("dice"), **out}
            ctx.store.put("runs", ctx.run_id, r)
        return out

    outs = parallel(ctx, "scouts", one, todo)
    n_ideas = sum(len(o.get("ideas") or []) for o in outs if o)
    n_sig = sum(len(o.get("signals") or []) for o in outs if o)
    ctx.bump({"signals": n_sig}, "scouts", {"in": len(todo), "ideas": n_ideas})
    ctx.log("scouts", f"{len(todo)} scouts: {n_ideas} ideas, {n_sig} signals")
    return {"done": ["scouts"]}


def dedupe_node(state: dict, ctx: Ctx) -> dict:
    run = ctx.store.run(ctx.run_id)
    if run.get("dedupe_done") or not run.get("scout_output"):
        return {"done": ["dedupe"]}
    cands = []
    for scout, o in run["scout_output"].items():
        sig_by_url = {s.get("url"): s for s in o.get("signals") or []}
        for idea in o.get("ideas") or []:
            sigs = [{"text": sig_by_url[u].get("text", ""), "url": u} for u in idea.get("signal_urls") or [] if u in sig_by_url]
            sigs += [{"text": ("direct: " if d.get("direct") else "") + d.get("text", ""), "url": d.get("url", "")}
                     for d in idea.get("direct_demand") or []]
            cands.append({**idea, "angle": o.get("angle"), "dice": o.get("dice"), "signals": sigs, "_scout": scout})
    existing = ctx.store.ideas()
    kept, dropped = dedupe_step.code_dedupe(cands, existing)
    if kept and not skip_if_over_budget(ctx, "dedupe model pass"):
        im = (ctx.store.config("idea_map") or {}).get("ideas", {})
        payload = {
            "candidates": [trim(c, ["title", "pitch", "nearest_existing", "difference", "category", "mechanic"]) for c in kept],
            "existing": [{"id": i, "title": d.get("title"), "pitch": (d.get("pitch") or "")[:110],
                          "category": d.get("category") or im.get(i, {}).get("category"),
                          "mechanic": d.get("mechanic") or im.get(i, {}).get("mechanic")} for i, d in existing.items()],
        }
        flags = llm.call("dedupe", payload, schemas.DEDUPE, node="dedupe", budget=ctx.budget).get("flags") or []
        ctx.save_usage()
        flagged = {f.get("candidate"): f for f in flags}
        still = []
        for c in kept:
            f = flagged.get(c.get("title"))
            if f:
                dropped.append({**c, "duplicate_of": f.get("duplicate_of"), "reason": f"{f.get('kind')}: {f.get('reason')}"})
            else:
                still.append(c)
        kept = still
    kept, over = dedupe_step.cap_batch(kept, config.MAX_NEW_IDEAS, config.MAX_BIG_IDEAS)
    for c in dropped:
        ctx.log("dedupe", f"Dropped '{c.get('title')}': {c.get('reason')} ({c.get('duplicate_of')})")
    have = {(d.get("run_id"), d.get("title")) for d in existing.values()}
    for c in kept:
        if (ctx.run_id, c.get("title")) in have:
            continue
        iid = ctx.store.new_idea_id(c["title"])
        doc = {k: c.get(k) for k in ["title", "pitch", "kind", "lane", "effort_weeks", "running_cost", "search_phrase",
                                     "known_rivals", "direct_demand", "nearest_existing", "difference", "category",
                                     "mechanic", "why_big", "v1", "signals", "angle", "dice"] if c.get(k) not in (None, "")}
        doc.update({"source": "scout", "stage": "generated", "run_id": ctx.run_id})
        ctx.store.create_idea(iid, doc, by=f"scout {c['_scout']}", note=c.get("angle") or "")
    ctx.patch_run({"dedupe_done": True})
    ctx.bump({"generated": len(kept), "duplicates": len(dropped), "over_cap": len(over)},
             "scouts", {"kept": len(kept)})
    ctx.log("dedupe", f"Kept {len(kept)}, dropped {len(dropped)} duplicates, {len(over)} over the cap")
    return {"done": ["dedupe"]}


def _critic_payload(d: dict, i: str, fill_only: bool = False) -> dict:
    p = {"id": i, **trim(d, ["title", "pitch", "kind", "lane", "effort_weeks", "running_cost", "search_phrase",
                             "signals", "known_rivals", "direct_demand", "why_big", "v1", "source"])}
    if is_michaels(d):
        p["notes"] = d.get("notes", "")
    if fill_only:
        p["fill_only"] = True
        p.update(trim(d, ["competitors", "wedge", "serp_note"]))
    return p


def apply_critique(ctx: Ctx, iid: str, r: dict, round_no: int, fill_only: bool = False) -> str:
    d = ctx.store.idea(iid)
    role = "critic1" if round_no == 1 else "critic2"
    if fill_only:
        patch = {k: r[k] for k in ("wedge", "serp_note", "competitors") if r.get(k)}
        ctx.store.update_idea(iid, patch, by=role, note="Filled wedge/serp_note for a carry-over" +
                              (f" (critic would {r.get('verdict')}: {r.get('kill_reason')})" if r.get("verdict") != "live" else ""))
        return "fill"
    verdict = r.get("verdict", "revise")
    stage, note = critic_outcome(d, verdict, r.get("kill_reason", ""), r.get("kill_category", ""), round_no)
    patch = {k: r[k] for k in CRITIC_FIELDS if r.get(k) not in (None, "", [])}
    patch["critique" if round_no == 1 else "critique2"] = {"at": now_iso(), "run_id": ctx.run_id, **r}
    patch["stage"] = stage
    if r.get("fixes"):
        patch["critic_fixes"] = r["fixes"]
    if stage == "killed":
        patch["stage_note"] = note
    elif note:
        patch["stage_note"] = note
    ctx.store.update_idea(iid, patch, by=role,
                          note=f"{verdict}" + (f": {r.get('kill_reason')}" if r.get("kill_reason") else "") +
                          (f" -> {stage}" if stage != {"live": "validated", "revise": "revising", "kill": "killed"}.get(verdict) else ""))
    ctx.bump({"killed": int(stage == "killed")}, role, {"in": 1, verdict: 1})
    return verdict


def critic1_node(state: dict, ctx: Ctx) -> dict:
    if skip_if_over_budget(ctx, "critic1"):
        return {"done": ["critic1"]}
    ideas = ctx.store.ideas()
    q = memory.queues(ideas, ctx.now())
    items = [(i, False) for i in q["critic1"]] + [(i, True) for i in q["critic1_fill"]]
    if not items:
        return {"done": ["critic1"]}
    taste = memory.taste_summary(ideas)

    def one(batch):
        payload = {"ideas": [_critic_payload(ideas[i], i, fill) for i, fill in batch], "taste": taste}
        out = llm.call("critic1", payload, schemas.CRITIC, node="critic1", budget=ctx.budget)
        fills = {i: f for i, f in batch}
        res = []
        for r in out.get("ideas") or []:
            if r.get("id") in fills:
                res.append(apply_critique(ctx, r["id"], r, 1, fills[r["id"]]))
        return res

    res = [v for b in parallel(ctx, "critic1", one, chunks(items, config.CRITIC_BATCH)) if b for v in b]
    ctx.log("critic1", f"{len(items)} ideas: " + ", ".join(f"{v} {res.count(v)}" for v in sorted(set(res))))
    return {"done": ["critic1"]}


def _problem(d: dict) -> dict:
    """What the reviser must fix: the newest critique's fixes, or a logistics blocker, or his parked reason."""
    lg = d.get("logistics") or {}
    c2, c1 = critique(d, "critique2"), critique(d, "critique")
    newest = max([x for x in (c1, c2, lg if lg.get("verdict") == "blocker" else {}) if x.get("at")],
                 key=lambda x: x["at"], default={})
    if newest is lg and lg:
        return {"from": "logistics", "blockers": lg.get("blockers"), "fixes": lg.get("fixes")}
    if newest:
        return {"from": "critic", "verdict": newest.get("verdict"), "kill_reason": newest.get("kill_reason"),
                "fixes": newest.get("fixes") or d.get("critic_fixes"), "fix_worked": newest.get("fix_worked")}
    dec = d.get("decision") or {}
    return {"from": "Michael (parked)", "fixes": [*(dec.get("reasons") or []), dec.get("text") or ""]}


def reviser_node(state: dict, ctx: Ctx) -> dict:
    if skip_if_over_budget(ctx, "reviser"):
        return {"done": ["reviser"]}
    ideas = ctx.store.ideas()
    q = memory.queues(ideas, ctx.now())
    for i in q["revive_parked"]:
        dec = ideas[i].get("decision") or {}
        reason = ", ".join([*(dec.get("reasons") or []), dec.get("text") or ""]).strip(", ")
        ctx.store.update_idea(i, {"stage": "revising", "revived_from_parked": True}, by="orchestrator",
                              note=f"Revived from parked: {reason}", allow_protected=True)
    ideas = ctx.store.ideas()
    todo = memory.queues(ideas, ctx.now())["reviser"]
    if not todo:
        return {"done": ["reviser"]}
    taste = memory.taste_summary(ideas)

    def one(batch):
        payload = {"ideas": [{"id": i, **trim(ideas[i], ["title", "pitch", "kind", "search_phrase", "lane", "source"]),
                              **({"notes": ideas[i].get("notes", "")} if is_michaels(ideas[i]) else {}),
                              "problem": _problem(ideas[i])} for i in batch], "taste": taste}
        out = llm.call("reviser", payload, schemas.REVISER, node="reviser", budget=ctx.budget)
        n = 0
        for r in out.get("ideas") or []:
            i = r.get("id")
            if i not in batch:
                continue
            d = ctx.store.idea(i)
            reason = "; ".join(str(x) for x in (_problem(d).get("fixes") or []) if x)[:400]
            rev = {"at": now_iso(), "by": "reviser", "change": r.get("change", ""), "reason": reason}
            if r.get("pivot") and rules.can_pivot(d):
                nid = ctx.store.new_idea_id(r["title"])
                ctx.store.create_idea(nid, {
                    "title": r["title"], "pitch": r["pitch"], "kind": r.get("kind") or d.get("kind"),
                    "search_phrase": r.get("search_phrase", ""), "lane": d.get("lane"), "parent": i,
                    "original_pitch": d.get("pitch"), "source": d.get("source", "scout"), "stage": "revising",
                    "run_id": ctx.run_id, "revisions": [{"v": 1, **rev}], "rev_count": 1, "angle": d.get("angle"),
                }, by="reviser", note=f"Pivot from {i}: {r.get('change', '')}")
                ctx.store.update_idea(i, {"stage": "killed", "stage_note": f"Pivoted to {nid}"}, by="reviser",
                                      note=f"Pivoted to {nid}")
                ctx.bump({"revised": 1}, "reviser", {"in": 1, "pivots": 1})
            else:
                rc = int(d.get("rev_count") or 0) + 1
                patch = {"title": r.get("title") or d.get("title"), "pitch": r.get("pitch") or d.get("pitch"),
                         "kind": r.get("kind") or d.get("kind"), "search_phrase": r.get("search_phrase") or d.get("search_phrase"),
                         "revisions": [*(d.get("revisions") or []), {"v": rc, **rev}], "rev_count": rc}
                if not d.get("original_pitch"):
                    patch["original_pitch"] = d.get("pitch")
                ctx.store.update_idea(i, patch, by="reviser", note=r.get("change", ""))
                ctx.bump({"revised": 1}, "reviser", {"in": 1, "revised": 1})
            n += 1
        return n

    n = sum(x or 0 for x in parallel(ctx, "reviser", one, chunks(todo, 10)))
    ctx.log("reviser", f"{n} of {len(todo)} revised")
    return {"done": ["reviser"]}


def critic2_node(state: dict, ctx: Ctx) -> dict:
    if skip_if_over_budget(ctx, "critic2"):
        return {"done": ["critic2"]}
    ideas = ctx.store.ideas()
    todo = memory.queues(ideas, ctx.now())["critic2"]
    if not todo:
        return {"done": ["critic2"]}

    def payload_for(i):
        d = ideas[i]
        r1 = critique(d, "critique")
        p = _critic_payload(d, i)
        p["change"] = ((d.get("revisions") or [{}])[-1]).get("change", "")
        p["round1"] = {**trim(r1, ["verdict", "kill_reason", "fixes", "wedge", "serp_note", "text"]),
                       "competitors": [trim(c, ["name", "url", "weakness"]) for c in (r1.get("competitors") or d.get("competitors") or [])][:6]}
        if (d.get("logistics") or {}).get("verdict") == "blocker":
            p["logistics_blocker"] = trim(d["logistics"], ["blockers", "fixes"])
        return p

    def one(batch):
        out = llm.call("critic2", {"ideas": [payload_for(i) for i in batch]}, schemas.CRITIC, node="critic2", budget=ctx.budget)
        return [apply_critique(ctx, r["id"], r, 2) for r in out.get("ideas") or [] if r.get("id") in batch]

    res = [v for b in parallel(ctx, "critic2", one, chunks(todo, config.CRITIC_BATCH)) if b for v in b]
    ctx.log("critic2", f"{len(todo)} ideas: " + ", ".join(f"{v} {res.count(v)}" for v in sorted(set(res))))
    return {"done": ["critic2"]}


def code_checks(d: dict, caches: dict) -> dict:
    if config.BACKEND == "demo":  # no network in demo runs
        return {"domains": {"demo.uk": "not found (likely free)"}, "urls": {}, "broken_urls": [], "licence_pages": []}
    domains = rdap.check_name(d.get("title", ""), caches["rdap"])
    urls = url_check.urls_in({k: d.get(k) for k in ("signals", "competitors", "data_sources", "build_plan")}, limit=25)
    status = url_check.check_all(urls, caches["urls"])
    pages = [licence_fetch.fetch(u, caches["pages"]) for u in licence_fetch.source_urls(d)]
    return {"domains": domains, "urls": {u: s for u, s in status.items()},
            "broken_urls": [u for u, s in status.items() if not (200 <= s < 400) and s not in (401, 403, 429)],
            "licence_pages": [p for p in pages if p.get("text")]}


def logistics_node(state: dict, ctx: Ctx) -> dict:
    if skip_if_over_budget(ctx, "logistics"):
        return {"done": ["logistics"]}
    ideas = ctx.store.ideas()
    todo = [i for i in memory.queues(ideas, ctx.now())["logistics"] if not is_protected(ideas[i])]
    if not todo:
        return {"done": ["logistics"]}
    caches = {"rdap": Cache("rdap", 7), "urls": Cache("urls", config.RIVAL_CACHE_DAYS),
              "pages": Cache("licence_pages", config.RIVAL_CACHE_DAYS)}
    checks = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, c in zip(todo, ex.map(lambda i: code_checks(ideas[i], caches), todo)):
            checks[i] = c
    for c in caches.values():
        c.save()
    ctx.log("logistics", f"Code checks done for {len(todo)} ideas "
            f"({sum(len(c['broken_urls']) for c in checks.values())} cited URLs didn't resolve)")

    def one(batch):
        payload = {"ideas": [{"id": i, **evidence(ideas[i]), "code_checks": {k: v for k, v in checks[i].items() if k != "urls"}}
                             for i in batch]}
        out = llm.call("logistics", payload, schemas.LOGISTICS, node="logistics", budget=ctx.budget)
        res = []
        for r in out.get("ideas") or []:
            i = r.get("id")
            if i not in batch:
                continue
            d = ctx.store.idea(i)
            lg = {"at": now_iso(), "run_id": ctx.run_id, **r,
                  "code_checks": {"domains": checks[i]["domains"], "broken_urls": checks[i]["broken_urls"]}}
            patch: dict = {"logistics": lg}
            v = r.get("verdict")
            if v == "fixable" and r.get("fixes"):
                patch["build_plan"] = (str(d.get("build_plan") or "") + "\nLogistics fixes: " + "; ".join(r["fixes"])).strip()
            stage, note = logistics_outcome(d, v, r.get("blockers") or [], bool(r.get("blocker_is_legal")),
                                            bool(r.get("blocker_has_no_workaround")))
            if stage:
                patch["stage"], patch["stage_note"] = stage, note
            ctx.store.update_idea(i, patch, by="logistics", note=f"Logistics check: {v}. {r.get('note', '')}".strip())
            ctx.bump({"logistics_checked": 1, "logistics_blocked": int(v == "blocker"), "killed": int(stage == "killed")},
                     "logistics", {"in": 1, v: 1})
            res.append(v)
        return res

    res = [v for b in parallel(ctx, "logistics", one, chunks(todo, config.LOGISTICS_BATCH)) if b for v in b]
    ctx.log("logistics", f"{len(res)} checked: " + ", ".join(f"{v} {res.count(v)}" for v in sorted(set(res))))
    return {"done": ["logistics"]}


def demand_node(state: dict, ctx: Ctx) -> dict:
    if skip_if_over_budget(ctx, "demand"):
        return {"done": ["demand"]}
    ideas = ctx.store.ideas()
    todo = [i for i in memory.queues(ideas, ctx.now())["demand"] if not is_protected(ideas[i])]
    if not todo:
        return {"done": ["demand"]}
    payload = {"ideas": [{"id": i, **trim(ideas[i], ["title", "pitch", "search_phrase", "demand_evidence", "signals",
                                                      "direct_demand", "audience_estimate"]),
                          "current_demand_score": (ideas[i].get("scores") or {}).get("demand")} for i in todo]}
    out = llm.call("demand", payload, schemas.DEMAND, node="demand", budget=ctx.budget)
    ctx.save_usage()
    res = []
    for r in out.get("ideas") or []:
        if r.get("id") not in todo:
            continue
        dh = {"at": now_iso(), "run_id": ctx.run_id, **{k: v for k, v in r.items() if k != "id"}}
        ctx.store.update_idea(r["id"], {"demand_hunt": dh}, by="demand",
                              note=f"Demand hunt: {r.get('verdict')} (suggested {int(r.get('suggested_demand_change') or 0):+d}). {r.get('note', '')}")
        ctx.bump({"demand_hunted": 1}, "demand", {"in": 1, r.get("verdict", "same"): 1})
        res.append(r.get("verdict"))
    ctx.log("demand", f"{len(res)} hunted: " + ", ".join(f"{v} {res.count(v)}" for v in sorted(set(res))))
    return {"done": ["demand"]}


def judge_targets(ideas: dict[str, dict], scope: str) -> list[str]:
    validated = sorted(i for i, d in ideas.items() if d.get("stage") == "validated")
    if scope == "validated":
        return validated
    board = [i for i, _ in leaderboard(ideas)]
    if scope == "all":
        return validated + board
    return validated + [i for i in board if has_new_evidence(ideas[i])]


def judge_node(state: dict, ctx: Ctx) -> dict:
    ideas = ctx.store.ideas()
    todo = [i for i in judge_targets(ideas, state.get("judge_scope", "changed")) if not is_protected(ideas[i])]
    if not todo:
        ctx.log("judge", "Nothing to score")
        return {"done": ["judge"]}
    ref = [i for i in top_n(ideas, 6) if i not in todo]
    base = {
        "reference_top6": [{"id": i, "title": ideas[i].get("title"), "pitch": (ideas[i].get("pitch") or "")[:200],
                            "lane": ideas[i].get("lane"), "scores": ideas[i].get("scores"), "total": ideas[i].get("total")}
                           for i in ref],
        "taste": memory.taste_summary(ideas),
        "live_results": memory.live_results(ctx.store, ideas, ctx.now()),
    }

    def entry(i):
        d = ideas[i]
        e = {"id": i, **evidence(d)}
        if d.get("scores") and d.get("stage") != "validated":
            e.update(previous_scores=d["scores"], previous_total=d.get("total"),
                     new_evidence_since_last_score=has_new_evidence(d))
        return e

    def one(batch):
        out = llm.call("judge", {**base, "to_score": [entry(i) for i in batch]}, schemas.JUDGE, node="judge", budget=ctx.budget)
        n = 0
        for r in out.get("ideas") or []:
            i = r.get("id")
            if i not in batch:
                continue
            d = ctx.store.idea(i)
            fo = r.get("famous_original") or d.get("famous_original") or "none"
            s, cap_notes = apply_famous_caps(r.get("scores") or {}, fo, bool(r.get("same_theme_version_exists")),
                                             bool(r.get("wedge_is_new_audience_or_mechanic")))
            carry = d.get("stage") != "validated" and d.get("scores")
            stab_notes = []
            if carry:
                s, stab_notes = apply_stability(d["scores"], s, has_new_evidence(d))
            t = total(s)
            notes = r.get("judge_notes", "")
            if cap_notes or stab_notes:
                notes += " [Code: " + "; ".join(cap_notes + stab_notes) + "]"
            patch = {"scores": s, "total": t, "judge_notes": notes, "judged_run_id": ctx.run_id, "stage": "judged"}
            if not d.get("famous_original"):
                patch["famous_original"] = fo
            if d.get("stage") in ("shortlisted",):
                patch["stage"] = "shortlisted"  # rerank decides
            ctx.store.update_idea(i, patch, by="judge",
                                  note=f"re-scored: {d.get('total')} → {t}" if carry else f"Judged {t}")
            ctx.bump({"judged": 1}, "judge", {"in": 1})
            n += 1
        if out.get("calibration_notes"):
            ctx.patch_run({"calibration_notes": out["calibration_notes"]})
        return n

    n = sum(x or 0 for x in parallel(ctx, "judge", one, chunks(todo, 8)))
    top = rerank(ctx)
    ctx.log("judge", f"Scored {n} of {len(todo)}. Top 6: " +
            ", ".join(f"{i} {ctx.store.idea(i).get('total')}" for i in top))
    return {"done": ["judge"]}


def factcheck_node(state: dict, ctx: Ctx) -> dict:
    if skip_if_over_budget(ctx, "factcheck"):
        return {"done": ["factcheck"]}
    ideas = ctx.store.ideas()
    todo = memory.factcheck_targets(ideas, ctx.now())
    if not todo:
        ctx.log("factcheck", "Top 3 already fact-checked within 30 days")
        return {"done": ["factcheck"]}
    out = llm.call("factcheck", {"ideas": [{"id": i, **evidence(ideas[i])} for i in todo]}, schemas.FACTCHECK,
                   node="factcheck", budget=ctx.budget)
    ctx.save_usage()
    for r in out.get("ideas") or []:
        i = r.get("id")
        if i not in todo:
            continue
        d = ctx.store.idea(i)
        s, notes = apply_fact_check(d.get("scores") or {}, r.get("score_changes"))
        t = total(s)
        patch = {"fact_check": {"at": now_iso(), "run_id": ctx.run_id, **trim(r, ["status", "summary", "checks"])},
                 "scores": s, "total": t,
                 "judge_notes": (d.get("judge_notes") or "") + f" Fact-check ({ctx.now():%-d %b}): {r.get('summary', '')}"}
        if r.get("missed_competitor_delivers_wedge"):
            st, note = rules.kill(d, "Fact-check: a competitor already delivers the wedge")
            patch.update(stage=st, stage_note=note)
        ctx.store.update_idea(i, patch, by="factcheck",
                              note=f"Fact-check {r.get('status')}" + (f": {d.get('total')} → {t}" if t != d.get("total") else ""))
        ctx.bump({"fact_checked": 1, "killed": int(patch.get("stage") == "killed")},
                 "factcheck", {"in": 1, r.get("status", "passed"): 1})
    rerank(ctx)
    ctx.log("factcheck", f"Checked {', '.join(todo)}")
    return {"done": ["factcheck"]}


def close_node(state: dict, ctx: Ctx) -> dict:
    ideas = ctx.store.ideas()
    top6 = top_n(ideas, 6)
    before = state.get("top6_before") or []
    run = ctx.store.run(ctx.run_id)
    stats = run.get("agent_stats") or {}
    ay = close_step.angle_yield(ideas, ctx.run_id, top6)
    if ay:
        stats["angle_yield"] = ay
    counts = run.get("counts") or {}
    counts["shortlisted"] = len(top6)
    ctx.patch_run({"agent_stats": stats, "counts": counts, "top": top6[:3],
                   "new_in_top6": len([i for i in top6 if i not in before]),
                   "usage": ctx.budget.to_dict()})
    run = ctx.store.run(ctx.run_id)
    run["_id"] = ctx.run_id
    ctx.patch_run({"summary": close_step.summary(run, ideas, top6), "lessons": close_step.lessons(run, ideas)})
    run = {**ctx.store.run(ctx.run_id), "_id": ctx.run_id, "status": "done"}
    text, worth = close_step.digest(ctx.store, run, ideas, before, ctx.now())
    config.OUT.mkdir(parents=True, exist_ok=True)
    (config.OUT / "digest.txt").write_text(text + "\n")
    sent = close_step.notify(text, ctx.now()) if worth else "not sent (nothing changed)"
    ctx.patch_run({"digest": text, "digest_sent": sent, "status": "done", "ended_at": now_iso(), "phase": None})
    ctx.log("close", f"Done. Digest {sent}.")
    print("\n" + text)
    return {"done": ["close"]}


NODES: dict[str, Callable[[dict, Ctx], dict]] = {
    "memory": memory_node, "trends": trends_node, "scouts": scouts_node, "dedupe": dedupe_node,
    "critic1": critic1_node, "reviser": reviser_node, "critic2": critic2_node, "logistics": logistics_node,
    "demand": demand_node, "judge": judge_node, "factcheck": factcheck_node, "close": close_node,
}
