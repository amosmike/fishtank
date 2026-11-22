"""Demo backend: simulated agents that return schema-valid answers for every role, with a short delay.

Everything except the model calls is the real pipeline (queues, rules, caps, stage transitions,
budgets, checkpoints), so a demo run shows exactly how ideas move. Used by `serve` and by the
end-to-end test. All content is fictional; every URL is example.com.
"""
from __future__ import annotations

import os
import random
import re
import time

from . import config
from .llm import Usage

EX = "https://example.com"
DELAY = float(os.environ.get("IDEA_LAB_DEMO_DELAY", 1.0))  # seconds per call, roughly; 0 in tests

POOL = [
    ("Queue Guess", "Guess how long today's queue is at famous UK attractions from a single photo.", "game"),
    ("Pitch Perfect Picnic", "Find the nearest picnic spot with shade, a bin and a toilet within 300 m.", "tool"),
    ("Hedge Fund", "An idle game where you grow a hedgerow and attract wildlife season by season.", "game"),
    ("Moth Night", "Log moths from your garden trap and see what's flying near you this week.", "tool"),
    ("Kerb Appeal", "Score your street's front gardens from photos and see the borough leaderboard.", "game"),
    ("Pothole Poker", "Bet chips on which reported pothole gets fixed first, using council open data.", "game"),
    ("Swim Window", "When the sea at your beach is warm, calm and clean enough for a dip.", "tool"),
    ("Recipe Roulette", "Spin three cupboard staples and get a dinner you can cook in 20 minutes.", "tool"),
    ("Canal Count", "Daily puzzle: count the locks between two canal towns from a map snippet.", "game"),
    ("Steam Days", "Every heritage railway running this weekend, filtered by distance and price.", "reference"),
    ("Pass the Parcel", "A multiplayer-by-link game where a parcel hops between friends until it pops.", "game"),
    ("Panto Planner", "Find this season's pantos by town, with ticket price bands and run dates.", "reference"),
    ("Mileage Mate", "Turn your running export into a map of every street you've covered.", "tool"),
    ("Brew Ratio", "Coffee ratio calculator for the kit you own, with a daily dial-in log.", "calculator"),
    ("Football Fives", "Five-a-side team balancer for group chats, with a fair-teams history.", "tool"),
    ("Words of the Week", "A ritual: five new English words a week, with one-minute daily recall.", "game"),
    ("Station Sprint", "How fast can you name the stations on a line, against today's leaderboard.", "game"),
    ("Pocket Planets", "A physics toy: drop planets into a tiny solar system and keep it stable.", "game"),
    ("Book Swap Map", "Little free libraries near you, with what people last left in them.", "reference"),
    ("Grit Bin Finder", "Where the nearest grit bin is, and whether the council says it's full.", "tool"),
]
CATS = ["daily puzzle game", "repeat-use tool", "reference lookup", "idle game", "money calculator", "community map"]
MECHS = ["guess", "planner", "lookup", "idle loop", "calculator", "leaderboard"]


def _rng(*parts) -> random.Random:
    return random.Random("|".join(str(p) for p in parts) + str(time.time_ns() // 10**9))


def _sleep() -> None:
    if DELAY > 0:
        time.sleep(DELAY * random.uniform(0.6, 1.6))


def respond(role: str, payload: dict, model: str, searches: int) -> tuple[dict, Usage]:
    _sleep()
    rng = _rng(role, id(payload))
    data = HANDLERS[role](payload, rng)
    tin, tout = config.EST_TOKENS.get(role, (20_000, 2_000))
    f = rng.uniform(0.15, 0.35)  # demo calls are modest so a full run fits inside the budget
    pin, pout = config.PRICE_PER_MTOK.get(model, (0, 0))
    s = rng.randint(max(0, searches // 3), searches) if searches else 0
    u = Usage(input_tokens=int(tin * f), output_tokens=int(tout * f), searches=s, calls=1)
    u.cost_usd = round(u.input_tokens / 1e6 * pin + u.output_tokens / 1e6 * pout + s * config.PRICE_PER_SEARCH, 4)
    return data, u


# ---- handlers ------------------------------------------------------------------------------------
def _classify(p, rng):
    return {"ideas": [{"id": i["id"], "category": rng.choice(CATS), "mechanic": rng.choice(MECHS)} for i in p["ideas"]]}


def _trends(p, rng):
    return {"trends": [{"trend": t, "why_now": "Fictional demo trend", "evidence_url": f"{EX}/trend/{k}",
                        "momentum": rng.choice(["rising", "peaking"]), "idea_seeds": [f"A {t.lower()} idea"]}
                       for k, t in enumerate(["Cosy outdoor hobbies", "Tiny daily rituals", "Group-chat games"])]}


def _scout(p, rng):
    taken = {re.split(r" — ", s)[0].strip().lower() for s in (p.get("memory") or {}).get("existing_ideas", [])}
    free = [x for x in POOL if x[0].lower() not in taken] or POOL
    picks = rng.sample(free, k=min(4, len(free)))
    angle = (p.get("brief") or {}).get("angle", "demo")
    signals = [{"text": f"Someone asking for '{t}' (demo signal)", "url": f"{EX}/signal/{_slug(t)}", "angle": angle}
               for t, _, _ in picks]
    ideas = []
    for t, pitch, kind in picks:
        big = rng.random() < 0.25
        ideas.append({
            "title": t, "pitch": pitch, "kind": kind, "lane": "big" if big else "week",
            "effort_weeks": 3 if big else 1, "running_cost": "under £5/month (demo)", "search_phrase": t.lower(),
            "signal_urls": [f"{EX}/signal/{_slug(t)}"], "known_rivals": [f"{EX}/rival/{_slug(t)}"],
            "direct_demand": [{"text": "A forum request for exactly this (demo)", "url": f"{EX}/ask/{_slug(t)}", "direct": True}],
            "nearest_existing": "none close", "difference": "Different user and mechanic (demo)",
            "category": rng.choice(CATS), "mechanic": rng.choice(MECHS),
            "why_big": "A content engine nobody else has (demo)" if big else "", "v1": "Two-week static v1 (demo)" if big else "",
        })
    return {"signals": signals, "ideas": ideas}


def _dedupe(p, rng):
    cands = p.get("candidates") or []
    flags = []
    if cands and rng.random() < 0.5:
        c = rng.choice(cands)
        flags.append({"candidate": c["title"], "duplicate_of": "an existing idea", "kind": "near-variant",
                      "reason": "Same mechanic, new theme (demo)"})
    return {"flags": flags}


def _critic(p, rng, round2=False):
    out = []
    for i in p["ideas"]:
        if i.get("fill_only"):
            v = "live"
        else:
            v = rng.choices(["live", "revise", "kill"], weights=[6, 3, 1.5] if round2 else [4, 4, 2])[0]
        kc = rng.choice(["incumbent", "no_wedge", "no_data"]) if v == "kill" else ""
        out.append({
            "id": i["id"], "verdict": v, "kill_reason": {"incumbent": "Free, good incumbent dominates (demo)",
                                                         "no_wedge": "No credible wedge (demo)",
                                                         "no_data": "No licensed data source (demo)"}.get(kc, ""),
            "kill_category": kc, "fix_worked": "The narrower angle works (demo)" if round2 else "",
            "fixes": ["Narrow to one region first (demo)", "Lead with a share loop (demo)"] if v == "revise" else [],
            "competitors": [{"name": "Rival site", "url": f"{EX}/rival/{i['id']}", "note": "Demo rival",
                             "weakness": "Slow on mobile, sign-up wall"}],
            "serp_note": "Small sites hold the long tail (demo)", "wedge": "One tap, no sign-up (demo)",
            "gap_note": "Weak incumbents (demo)", "demand_evidence": "Two direct asks (demo)",
            "audience_estimate": "Tens of thousands (demo estimate)", "data_sources": f"{EX}/open-data (demo, OGL-style)",
            "build_plan": "Static page + JSON built nightly (demo)", "traffic_plan": "Share loop + two communities (demo)",
            "monetisation": "Display ads later", "risks": "Data freshness (demo)", "lane": i.get("lane") or "week",
            "effort_weeks": 1, "running_cost": "£0-5/month (demo)", "famous_original": "none",
        })
    return {"ideas": out}


def _reviser(p, rng):
    out = []
    for i in p["ideas"]:
        out.append({"id": i["id"], "title": i.get("title", ""), "pitch": (i.get("pitch") or "") + " Narrowed to one region first.",
                    "kind": i.get("kind") or "game", "search_phrase": i.get("search_phrase") or "",
                    "change": "Narrowed to one region; added a share line (demo)", "pivot": rng.random() < 0.08})
    return {"ideas": out}


def _logistics(p, rng):
    out = []
    for i in p["ideas"]:
        v = rng.choices(["go", "fixable", "blocker"], weights=[5, 4, 1])[0]
        out.append({"id": i["id"], "verdict": v,
                    "data": {"source": "Demo open dataset", "url": f"{EX}/data", "licence": "OGL-style (demo)",
                             "update": "weekly", "curation": "1 hour/week"},
                    "stack": "Static site + scheduled JSON build", "free_tier_limit": "None at 10k/day (demo)",
                    "cost_monthly": {"at_1k": "£0", "at_10k": "£0-5", "price_url": f"{EX}/pricing"},
                    "effort_days": rng.choice([3, 5, 8, 12]), "upkeep_hours_week": rng.choice([0.5, 1, 2]),
                    "riskiest_part": "Keeping the data fresh (demo)", "name_check": "No clash found (demo)",
                    "domain_check": ".uk looks free (demo)", "ad_policy": "Fine for AdSense (demo)",
                    "compliance": "Cookie consent for ads (demo)",
                    "blockers": ["Data licence is non-commercial (demo)"] if v == "blocker" else [],
                    "blocker_is_legal": False, "blocker_has_no_workaround": v == "blocker" and rng.random() < 0.5,
                    "fixes": ["Swap to an OGL source (demo)"] if v == "fixable" else [], "note": f"Logistics: {v} (demo)"})
    return {"ideas": out}


def _demand(p, rng):
    out = []
    for i in p["ideas"]:
        v = rng.choice(["stronger", "same", "weaker"])
        out.append({"id": i["id"], "direct": [{"text": "A direct request (demo)", "url": f"{EX}/ask/{i['id']}", "date": "2026-11"}],
                    "proxies": [{"text": "Autocomplete phrasing (demo)", "url": f"{EX}/proxy"}], "against": [],
                    "verdict": v, "suggested_demand_change": {"stronger": 1, "same": 0, "weaker": -1}[v],
                    "note": f"Demand looks {v} (demo)"})
    return {"ideas": out}


def _judge(p, rng):
    out = []
    for e in p["to_score"]:
        r = random.Random(e["id"] + str(rng.random()))
        s = {k: r.choice([2, 3, 3, 4, 4, 5]) for k in ("demand", "audience", "gap", "build", "share", "ads", "fun")}
        out.append({"id": e["id"], "scores": s, "famous_original": "none", "same_theme_version_exists": False,
                    "wedge_is_new_audience_or_mechanic": False,
                    "judge_notes": f"Lane {e.get('lane') or 'week'}. Demo scores from the evidence given."})
    return {"ideas": out, "calibration_notes": "Too early to tell (demo)."}


def _factcheck(p, rng):
    return {"ideas": [{"id": e["id"], "status": rng.choice(["passed", "passed", "flagged"]), "summary": "Demo fact-check.",
                       "checks": [{"claim": "No strong competitor", "ok": True, "url": f"{EX}/check", "note": "demo"}],
                       "score_changes": {"demand": 0, "gap": rng.choice([0, 0, -1])},
                       "missed_competitor_delivers_wedge": False} for e in p["ideas"]]}


def _slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


HANDLERS = {
    "classify": _classify, "trends": _trends, "scout": _scout, "dedupe": _dedupe,
    "critic1": _critic, "critic2": lambda p, r: _critic(p, r, round2=True), "reviser": _reviser,
    "logistics": _logistics, "demand": _demand, "judge": _judge, "factcheck": _factcheck,
}
