"""JSON schemas for each role's output (used as structured-output constraints on both backends)."""
from __future__ import annotations


def obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "additionalProperties": False,
            "required": list(props) if required is None else required}


def arr(items: dict) -> dict:
    return {"type": "array", "items": items}


S = {"type": "string"}
I = {"type": "integer"}
N = {"type": "number"}
B = {"type": "boolean"}


def enum(*vals: str) -> dict:
    return {"type": "string", "enum": list(vals)}


LINK = obj({"text": S, "url": S})
COMPETITOR = obj({"name": S, "url": S, "note": S, "weakness": S})
SCORES = obj({k: I for k in ("demand", "audience", "gap", "build", "share", "ads", "fun")})

TRENDS = obj({"trends": arr(obj({
    "trend": S, "why_now": S, "evidence_url": S, "momentum": enum("rising", "peaking", "fading"),
    "idea_seeds": arr(S)}))})

SCOUT = obj({
    "signals": arr(obj({"text": S, "url": S, "angle": S})),
    "ideas": arr(obj({
        "title": S, "pitch": S, "kind": enum("game", "tool", "reference", "calculator"),
        "lane": enum("week", "big"), "effort_weeks": N, "running_cost": S, "search_phrase": S,
        "signal_urls": arr(S), "known_rivals": arr(S),
        "direct_demand": arr(obj({"text": S, "url": S, "direct": B})),
        "nearest_existing": S, "difference": S, "category": S, "mechanic": S,
        "why_big": S, "v1": S,
    })),
})

DEDUPE = obj({"flags": arr(obj({
    "candidate": S, "duplicate_of": S, "kind": enum("duplicate", "near-variant", "same-batch"), "reason": S}))})

CLASSIFY = obj({"ideas": arr(obj({"id": S, "category": S, "mechanic": S}))})

CRITIC = obj({"ideas": arr(obj({
    "id": S, "verdict": enum("live", "revise", "kill"), "kill_reason": S,
    "kill_category": enum("", "incumbent", "no_wedge", "no_data", "policy"),
    "fix_worked": S, "fixes": arr(S), "competitors": arr(COMPETITOR), "serp_note": S, "wedge": S,
    "gap_note": S, "demand_evidence": S, "audience_estimate": S, "data_sources": S, "build_plan": S,
    "traffic_plan": S, "monetisation": S, "risks": S, "lane": enum("week", "big"), "effort_weeks": N,
    "running_cost": S, "famous_original": S,
}))})

REVISER = obj({"ideas": arr(obj({
    "id": S, "title": S, "pitch": S, "kind": enum("game", "tool", "reference", "calculator"),
    "search_phrase": S, "change": S, "pivot": B}))})

LOGISTICS = obj({"ideas": arr(obj({
    "id": S, "verdict": enum("go", "fixable", "blocker"),
    "data": obj({"source": S, "url": S, "licence": S, "update": S, "curation": S}),
    "stack": S, "free_tier_limit": S,
    "cost_monthly": obj({"at_1k": S, "at_10k": S, "price_url": S}),
    "effort_days": N, "upkeep_hours_week": N, "riskiest_part": S, "name_check": S, "domain_check": S,
    "ad_policy": S, "compliance": S, "blockers": arr(S), "blocker_is_legal": B,
    "blocker_has_no_workaround": B, "fixes": arr(S), "note": S}))})

DEMAND = obj({"ideas": arr(obj({
    "id": S, "direct": arr(obj({"text": S, "url": S, "date": S})), "proxies": arr(LINK),
    "against": arr(LINK), "verdict": enum("stronger", "same", "weaker"),
    "suggested_demand_change": I, "note": S}))})

JUDGE = obj({
    "ideas": arr(obj({
        "id": S, "scores": SCORES, "famous_original": S, "same_theme_version_exists": B,
        "wedge_is_new_audience_or_mechanic": B, "judge_notes": S})),
    "calibration_notes": S,
})

FACTCHECK = obj({"ideas": arr(obj({
    "id": S, "status": enum("passed", "flagged"), "summary": S,
    "checks": arr(obj({"claim": S, "ok": B, "url": S, "note": S})),
    "score_changes": obj({"demand": I, "gap": I}),
    "missed_competitor_delivers_wedge": B}))})
