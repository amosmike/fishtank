"""The playbook's deterministic rules, as pure functions. Everything here is unit-tested.

Nothing in this module calls a model, reads files or touches the clock (callers pass times in).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

# ---- stages -----------------------------------------------------------------------------------
STAGES = ["submitted", "generated", "validated", "revising", "killed", "judged", "shortlisted",
          "selected", "parked", "rejected", "building", "hosted"]
# Michael's decisions. Code never changes an idea in these stages except to append history.
PROTECTED_STAGES = frozenset({"selected", "parked", "rejected", "building", "hosted"})
LEADERBOARD_STAGES = frozenset({"judged", "shortlisted"})


def is_protected(idea: dict) -> bool:
    # Keyed on stage, not on `decision`: a parked idea revived under the approved exception keeps
    # its old decision record but is back in the pipeline.
    return idea.get("stage") in PROTECTED_STAGES


def is_michaels(idea: dict) -> bool:
    return idea.get("source") == "michael"


# ---- scoring ----------------------------------------------------------------------------------
WEIGHTS = {"demand": 3, "audience": 2, "gap": 2, "build": 2, "share": 2, "ads": 1, "fun": 1}
MAX_WEIGHTED = 5 * sum(WEIGHTS.values())  # 65


def clamp_score(v) -> int:
    try:
        return max(0, min(5, int(round(float(v)))))
    except (TypeError, ValueError):
        return 0


def total(scores: dict) -> int:
    """round(sum(w*s)/65*100), rounding halves up (matches every stored total in the lab)."""
    raw = sum(WEIGHTS[k] * clamp_score(scores.get(k, 0)) for k in WEIGHTS) / MAX_WEIGHTED * 100
    return int(math.floor(raw + 0.5 + 1e-9))


def has_famous_original(famous_original: str | None) -> bool:
    if not famous_original:
        return False
    return not str(famous_original).strip().lower().startswith("none")


def apply_famous_caps(scores: dict, famous_original: str | None, same_theme_exists: bool,
                      wedge_exempt: bool) -> tuple[dict, list[str]]:
    """FAMOUS-ORIGINAL RULE (v10). Returns (capped scores, notes on caps applied).

    gap <= 2 unless the wedge is a different audience the original can't serve or a different way
    of playing (`wedge_exempt`); gap <= 1 if a same-theme version already exists (always);
    share <= 4 whenever a famous original applies.
    """
    s = {k: clamp_score(scores.get(k, 0)) for k in WEIGHTS}
    notes: list[str] = []
    if not has_famous_original(famous_original):
        return s, notes
    gap_cap = 1 if same_theme_exists else (5 if wedge_exempt else 2)
    if s["gap"] > gap_cap:
        notes.append(f"gap {s['gap']}->{gap_cap} (famous original: {famous_original})")
        s["gap"] = gap_cap
    if s["share"] > 4:
        notes.append(f"share {s['share']}->4 (famous original share grid)")
        s["share"] = 4
    return s, notes


def apply_stability(prev: dict | None, new: dict, new_evidence: bool) -> tuple[dict, list[str]]:
    """STABILITY: without new evidence since the last score, no criterion moves by more than 1."""
    s = {k: clamp_score(new.get(k, 0)) for k in WEIGHTS}
    if not prev or new_evidence:
        return s, []
    notes = []
    for k in WEIGHTS:
        p = clamp_score(prev.get(k, 0))
        lo, hi = p - 1, p + 1
        if s[k] < lo or s[k] > hi:
            v = max(lo, min(hi, s[k]))
            notes.append(f"{k} {s[k]}->{v} (stability: no new evidence)")
            s[k] = v
    return s, notes


EVIDENCE_FIELDS = ("critique", "critique2", "demand_hunt", "fact_check", "logistics")


def newest_evidence_at(idea: dict) -> datetime | None:
    best = None
    for f in EVIDENCE_FIELDS:
        v = idea.get(f)
        at = _ts(v.get("at")) if isinstance(v, dict) else None
        if at and (best is None or at > best):
            best = at
    for r in idea.get("revisions") or []:
        at = _ts(r.get("at"))
        if at and (best is None or at > best):
            best = at
    return best


def last_scored_at(idea: dict) -> datetime | None:
    best = None
    for h in idea.get("history") or []:
        if h.get("by", "").startswith("judge") or "fact" in h.get("by", ""):
            at = _ts(h.get("at"))
            if at and (best is None or at > best):
                best = at
    return best


def has_new_evidence(idea: dict) -> bool:
    ev, sc = newest_evidence_at(idea), last_scored_at(idea)
    return sc is None or (ev is not None and ev > sc)


def apply_fact_check(scores: dict, changes: dict | None) -> tuple[dict, list[str]]:
    """Fact-check may only LOWER scores, by at most 2 per criterion, never below 0."""
    s = {k: clamp_score(scores.get(k, 0)) for k in WEIGHTS}
    notes = []
    for k, delta in (changes or {}).items():
        if k not in WEIGHTS:
            continue
        try:
            d = int(delta)
        except (TypeError, ValueError):
            continue
        d = max(-2, min(0, d))
        if d:
            v = max(0, s[k] + d)
            notes.append(f"{k} {s[k]}->{v} (fact-check)")
            s[k] = v
    return s, notes


# ---- critic verdicts and the revision cap -------------------------------------------------------
MAX_REVISIONS = 2


def critic_outcome(idea: dict, verdict: str, kill_reason: str = "", kill_category: str = "",
                   round_no: int = 1) -> tuple[str, str]:
    """Map a critic verdict to (new stage, stage_note), applying Michael's rules and the cap.

    - Michael's ideas are never killed on the first pass except for policy/legal problems.
    - A "revise" on an idea that already had 2 revisions becomes a kill.
    """
    verdict = (verdict or "").lower()
    if verdict == "live":
        return "validated", ""
    if verdict == "kill":
        if is_michaels(idea) and round_no == 1 and kill_category != "policy":
            return "revising", f"Critic would kill ({kill_reason}); Michael's idea goes to the reviser instead"
        return "killed", kill_reason or "Killed by critic"
    # revise
    if int(idea.get("rev_count") or 0) >= MAX_REVISIONS:
        return "killed", f"Still not working after {MAX_REVISIONS} revisions: {kill_reason or 'critic asked for a third revision'}"
    return "revising", ""


def logistics_outcome(idea: dict, verdict: str, blockers: list[str], legal: bool,
                      no_workaround: bool) -> tuple[str | None, str]:
    """Map a logistics verdict to (new stage or None for unchanged, note).

    "blocker" kills only when no workaround exists; otherwise it's a revision (which counts toward
    the cap). Michael's ideas are never killed on logistics except for legal problems.
    """
    if verdict != "blocker":
        return None, ""
    reason = "; ".join(blockers)[:200] or "blocker"
    if is_michaels(idea):
        if legal:
            return "killed", f"Logistics (legal): {reason}"
        return "revising", f"Logistics: {reason}"
    if no_workaround or legal:
        return "killed", f"Logistics: {reason}"
    if int(idea.get("rev_count") or 0) >= MAX_REVISIONS:
        return "killed", f"Still not working after {MAX_REVISIONS} revisions: logistics: {reason}"
    return "revising", f"Logistics: {reason}"


# ---- leaderboard ------------------------------------------------------------------------------
def leaderboard(ideas: dict[str, dict]) -> list[tuple[str, dict]]:
    """Scored, undecided ideas, best first. Ties keep a stable order by id."""
    rows = [(i, d) for i, d in ideas.items()
            if d.get("stage") in LEADERBOARD_STAGES and d.get("total") is not None]
    rows.sort(key=lambda r: (-int(r[1]["total"]), r[0]))
    return rows


def top_n(ideas: dict[str, dict], n: int = 6) -> list[str]:
    return [i for i, _ in leaderboard(ideas)[:n]]


def stale(at: str | None, now: datetime, max_age_days: int) -> bool:
    t = _ts(at)
    return t is None or now - t > timedelta(days=max_age_days)


def _ts(s) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
