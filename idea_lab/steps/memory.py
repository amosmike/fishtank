"""Memory: the brief every agent gets, and the work queues. Pure code over the JSON store.

Queues are computed from the data every time (not carried in graph state), so every node is
idempotent: re-running a node after a crash only picks up ideas it hasn't handled yet.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta

from .. import config
from ..rules import PROTECTED_STAGES, has_new_evidence, is_michaels, leaderboard, stale, top_n
from ..store import Store, parse_ts

REVIVE_REASONS = ("needs a better angle", "needs a data source", "needs data source")


# ---- helpers ------------------------------------------------------------------------------------
def _at(v) -> datetime | None:
    return parse_ts(v.get("at")) if isinstance(v, dict) else None


def last_revision_at(idea: dict) -> datetime | None:
    revs = idea.get("revisions") or []
    return max((t for t in (parse_ts(r.get("at")) for r in revs) if t), default=None)


def last_verdict_at(idea: dict) -> datetime | None:
    """Newest critic verdict or logistics blocker on the idea."""
    ts = [_at(idea.get("critique")), _at(idea.get("critique2"))]
    lg = idea.get("logistics") or {}
    if lg.get("verdict") == "blocker":
        ts.append(_at(lg))
    return max((t for t in ts if t), default=None)


def needs_revision(idea: dict) -> bool:
    """In "revising" and the newest verdict came after the newest revision."""
    if idea.get("stage") != "revising":
        return False
    rev, ver = last_revision_at(idea), last_verdict_at(idea)
    return rev is None or (ver is not None and ver > rev)


def needs_critic2(idea: dict) -> bool:
    """In "revising" with a revision that no critic has looked at yet.

    A tie counts as unchecked: the reviser only ever runs after the verdict it answers, so a
    revision stamped at the same instant is the newer of the two."""
    if idea.get("stage") != "revising":
        return False
    rev, ver = last_revision_at(idea), last_verdict_at(idea)
    return rev is not None and (ver is None or rev >= ver)


def parked_revivable(idea: dict) -> bool:
    d = idea.get("decision") or {}
    if idea.get("stage") != "parked" or idea.get("revived_from_parked"):
        return False
    words = " ".join([*(d.get("reasons") or []), d.get("text") or ""]).lower()
    return any(r in words for r in REVIVE_REASONS)


# ---- queues ------------------------------------------------------------------------------------
def queues(ideas: dict[str, dict], now: datetime) -> dict[str, list[str]]:
    top6 = top_n(ideas, 6)
    q: dict[str, list[str]] = {}
    q["critic1"] = sorted([i for i, d in ideas.items() if d.get("stage") == "submitted"]) + \
        sorted([i for i, d in ideas.items() if d.get("stage") == "generated"])
    q["critic1_fill"] = [i for i in top_n(ideas, 12)
                         if not ideas[i].get("wedge") or not ideas[i].get("serp_note")]
    q["reviser"] = sorted(i for i, d in ideas.items() if needs_revision(d))
    q["revive_parked"] = sorted(i for i, d in ideas.items() if parked_revivable(d))
    q["critic2"] = sorted(i for i, d in ideas.items() if needs_critic2(d))
    q["validated"] = sorted(i for i, d in ideas.items() if d.get("stage") == "validated")
    q["logistics"] = logistics_targets(ideas, now, top6)
    q["demand"] = demand_targets(ideas, now, top6)
    q["changed_carryovers"] = [i for i, _ in leaderboard(ideas) if has_new_evidence(ideas[i])]
    q["top6"] = top6
    return q


def logistics_targets(ideas: dict[str, dict], now: datetime, top6: list[str]) -> list[str]:
    def due(d: dict) -> bool:
        lg = d.get("logistics")
        if not lg:
            return True
        rev = last_revision_at(d)
        lg_at = _at(lg)
        return stale(lg.get("at"), now, config.LOGISTICS_MAX_AGE_DAYS) or bool(rev and lg_at and rev > lg_at)

    heading = [i for i, d in ideas.items() if d.get("stage") == "validated" and due(d)]
    heading.sort(key=lambda i: (ideas[i].get("lane") != "big", i))
    carry = [i for i in top6 if i not in heading and due(ideas[i])]
    return (heading + carry)[: config.LOGISTICS_TARGETS]


def demand_targets(ideas: dict[str, dict], now: datetime, top6: list[str]) -> list[str]:
    def hunted_recently(d: dict) -> bool:
        dh = d.get("demand_hunt")
        return bool(dh) and not stale(dh.get("at"), now, config.DEMAND_HUNT_MAX_AGE_DAYS)

    first = [i for i in top6 if (ideas[i].get("scores") or {}).get("demand", 5) <= 3 and not hunted_recently(ideas[i])]
    then = [i for i, d in sorted(ideas.items()) if d.get("stage") == "validated" and not d.get("demand_hunt")]
    out: list[str] = []
    for i in first + then:
        if i not in out:
            out.append(i)
    return out[: config.DEMAND_TARGETS]


def factcheck_targets(ideas: dict[str, dict], now: datetime) -> list[str]:
    out = []
    for i in top_n(ideas, config.FACTCHECK_TOP):
        fc = ideas[i].get("fact_check")
        rev = last_revision_at(ideas[i])
        if not fc or stale(fc.get("at"), now, config.FACTCHECK_MAX_AGE_DAYS) or (rev and _at(fc) and rev > _at(fc)):
            out.append(i)
    return out


# ---- brief ---------------------------------------------------------------------------------------
def uk_season(d: date) -> str:
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer"}.get(d.month, "autumn")


def taste_summary(ideas: dict[str, dict]) -> dict:
    """Michael's decisions and words, verbatim. This outranks everything else in every brief."""
    likes, rejects, parked, own = [], [], [], []
    for i, d in sorted(ideas.items()):
        dec = d.get("decision") or {}
        line = {"idea": d.get("title") or i, "reasons": dec.get("reasons") or [], "his_words": dec.get("text") or ""}
        if dec.get("status") == "selected":
            likes.append(line)
        elif dec.get("status") == "rejected":
            rejects.append(line)
        elif dec.get("status") == "parked":
            parked.append(line)
        if is_michaels(d):
            own.append({"idea": d.get("title") or i, "pitch": d.get("pitch", ""), "his_notes": d.get("notes", "")})
    return {
        "michael_picked": likes, "michael_rejected": rejects, "michael_parked": parked, "michaels_own_ideas": own,
        "standing_rules": [
            "No reskins of famous free daily games (Wordle, Connections, Worldle…) with a UK theme on top.",
            "Check a name and its domain for existing games before buying.",
            "Sites are judged on return visits at weeks 4 and 8, not early ad money.",
            "No paid advertising before a site has ads.",
        ],
    }


def idea_map(ideas: dict[str, dict], classes: dict[str, dict]) -> dict:
    cat, mech = Counter(), Counter()
    for i, d in ideas.items():
        c = d.get("category") or (classes.get(i) or {}).get("category") or d.get("kind") or "unknown"
        m = d.get("mechanic") or (classes.get(i) or {}).get("mechanic") or "unknown"
        cat[_clean_label(c)] += 1
        mech[_clean_label(m)] += 1
    thin = [c for c, n in cat.items() if n <= 2]
    return {"categories": dict(cat.most_common()), "mechanics": dict(mech.most_common(40)),
            "thin_or_empty_categories": sorted(thin)}


def _clean_label(s: str) -> str:
    s = str(s).lower()
    for tag in ("(empty)", "(thin)"):
        s = s.replace(tag, "")
    return s.strip(" ,")


def live_results(store: Store, ideas: dict[str, dict], now: datetime) -> list[str]:
    out = []
    for sid, s in store.all("sites").items():
        weeks = s.get("weeks") or []
        launched = parse_ts((s.get("launched") or "") + "T00:00:00Z")
        age = (now - launched).days // 7 if launched else None
        idea = ideas.get(s.get("idea_id") or "", {})
        last = weeks[-1] if weeks else {}
        visits = sum(int(w.get("visits") or 0) for w in weeks)
        earn = sum(float(w.get("earnings") or 0) for w in weeks)
        line = (f"{s.get('name', sid)} (idea {s.get('idea_id')}, scored {idea.get('total')}, scores {idea.get('scores')}), "
                f"status {s.get('status')}, live {age} weeks, {len(weeks)} weeks logged")
        if weeks:
            line += (f": latest quiet-day avg {last.get('quiet_avg')}/day, {visits} visits total, "
                     f"£{earn:.2f} earned, top referrer {last.get('referrer') or 'n/a'}, note: {last.get('note') or ''}")
        else:
            line += ": no numbers logged yet, too early to draw lessons"
        out.append(line)
    return out


def recent_kills(ideas: dict[str, dict], n: int = 15) -> list[dict]:
    killed = [(d.get("updated_at") or "", i, d) for i, d in ideas.items() if d.get("stage") == "killed"]
    killed.sort(reverse=True)
    return [{"idea": d.get("title") or i, "why": (d.get("stage_note") or "")[:160]} for _, i, d in killed[:n]]


def brief(store: Store, now: datetime, for_role: str = "scout") -> dict:
    ideas = store.ideas()
    runs = store.runs()
    classes = (store.config("idea_map") or {}).get("ideas", {})
    last = runs[-1] if runs else {}
    b = {
        "date": now.date().isoformat(), "uk_season": uk_season(now.date()),
        "seasonal_rule": f"Seasonal ideas must target a moment 3+ months away (after {(now + timedelta(days=90)).date():%d %b %Y}).",
        "taste": taste_summary(ideas),
        "live_results": live_results(store, ideas, now),
        "last_run_lessons": last.get("lessons") or [],
    }
    if for_role == "scout":
        b["idea_map"] = idea_map(ideas, classes)
        from ..retrieval import rag
        if rag.available():  # every title (never re-propose these); the relevant ones come in full via related_ideas
            b["existing_titles"] = [d.get("title") or i for i, d in sorted(ideas.items())]
        else:
            b["existing_ideas"] = [f"{d.get('title') or i} — {(d.get('pitch') or '')[:90]}" for i, d in sorted(ideas.items())]
        b["recent_critic_kills"] = recent_kills(ideas)
        b["stopped_sites_no_variants"] = [s.get("name") for s in store.all("sites").values() if s.get("status") == "stopped"]
    return b


def saturated(ideas: dict[str, dict]) -> bool:
    """SATURATION GUARD: 8+ ideas in revising -> discovery behaves as maintenance."""
    return sum(1 for d in ideas.values() if d.get("stage") == "revising") >= 8


__all__ = ["queues", "brief", "factcheck_targets", "needs_revision", "needs_critic2", "parked_revivable",
           "taste_summary", "idea_map", "saturated", "PROTECTED_STAGES"]
