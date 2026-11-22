"""Close: counts, angle yield, top 6 movement, lessons and the digest. No model."""
from __future__ import annotations

import os
import urllib.request
from collections import Counter
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ..rules import top_n
from ..store import Store, parse_ts

LONDON = ZoneInfo("Europe/London")


def angle_yield(ideas: dict[str, dict], run_id: str, top6: list[str]) -> dict:
    out: dict[str, dict] = {}
    for i, d in ideas.items():
        if d.get("run_id") != run_id or d.get("source") == "michael":
            continue
        a = d.get("angle") or "unknown"
        y = out.setdefault(a, {"ideas": 0, "survived_critics": 0, "shortlisted": 0})
        y["ideas"] += 1
        if d.get("stage") in ("validated", "judged", "shortlisted") or any(
                h.get("stage") == "validated" for h in d.get("history") or []):
            y["survived_critics"] += 1
        if i in top6:
            y["shortlisted"] += 1
    return out


def kills_this_run(ideas: dict[str, dict], started: datetime) -> list[tuple[str, str]]:
    out = []
    for i, d in ideas.items():
        for h in d.get("history") or []:
            t = parse_ts(h.get("at"))
            if h.get("stage") == "killed" and t and t >= started and d.get("stage") == "killed":
                out.append((d.get("title") or i, d.get("stage_note") or h.get("note") or ""))
                break
    return out


def live_sites_line(store: Store, now: datetime) -> list[str]:
    lines = []
    local = now.astimezone(LONDON)
    for sid, s in store.all("sites").items():
        if s.get("status") != "live":
            continue
        weeks = s.get("weeks") or []
        name = s.get("name") or sid
        if local.weekday() == 0:  # Monday: last week's numbers due
            last_mon = (local - timedelta(days=7)).date()
            last_mon = last_mon - timedelta(days=last_mon.weekday())
            if not any(w.get("week_start") == last_mon.isoformat() for w in weeks):
                lines.append(f"Log {name}'s week of {last_mon:%-d %b}.")
        launched = parse_ts((s.get("launched") or "") + "T00:00:00Z")
        if launched:
            age = (now - launched).days
            for wk, days in ((4, 28), (8, 56)):
                if days <= age < days + 7:
                    q = weeks[-1].get("quiet_avg") if weeks else None
                    if q is None:
                        verdict = "no numbers logged, so no verdict"
                    elif q >= 50:
                        verdict = "keep going"
                    elif q < 10:
                        verdict = "push once more" if wk == 4 else "stop putting time in (leave it up)"
                    else:
                        verdict = "in between: keep watching"
                    lines.append(f"{name} is at its week-{wk} checkpoint: quiet-day average {q if q is not None else 'n/a'}/day, {verdict}.")
    return lines


def digest(store: Store, run: dict, ideas: dict[str, dict], top6_before: list[str], now: datetime) -> tuple[str, bool]:
    """Returns (digest text, worth sending)."""
    top6 = top_n(ideas, 6)
    started = parse_ts(run.get("started_at")) or now
    entered = [i for i in top6 if i not in top6_before]
    left = [i for i in top6_before if i not in top6]
    kills = kills_this_run(ideas, started)
    fails = [e["msg"] for e in run.get("log") or [] if e.get("agent") == "error"]
    sites = live_sites_line(store, now)
    usage = (run.get("usage") or {}).get("total") or {}

    def name(i):
        d = ideas.get(i, {})
        return f"{d.get('title') or i} ({d.get('total')})"

    lines = [f"Idea Lab — {run.get('_id') or ''} ({run.get('run_type')}, {run.get('status')})"]
    lines.append("Top 3: " + (", ".join(name(i) for i in top6[:3]) or "none"))
    if entered:
        lines.append("Entered the top 6: " + ", ".join(name(i) for i in entered))
    if left:
        lines.append("Left the top 6: " + ", ".join(name(i) for i in left))
    if kills:
        lines.append("Killed: " + "; ".join(f"{t} ({n[:80]})" for t, n in kills[:5]))
    if fails:
        lines.append("Failed: " + "; ".join(f[:120] for f in fails[:3]))
    if sites:
        lines.append("LIVE SITES: " + " ".join(sites))
    lines.append(f"Cost: {usage.get('tokens', 0):,} tokens, {usage.get('searches', 0)} searches, "
                 f"{usage.get('calls', 0)} model calls, ~${usage.get('cost_usd', 0):.2f}.")
    changed = bool(entered or left or kills or fails or any(
        h for d in ideas.values() for h in d.get("history") or []
        if (t := parse_ts(h.get("at"))) and t >= started and h.get("by") == "judge"))
    return "\n".join(lines), bool(changed or sites)


def summary(run: dict, ideas: dict[str, dict], top6: list[str]) -> str:
    c = run.get("counts") or {}
    s1 = (f"{c.get('generated', 0)} new ideas, {c.get('killed', 0)} killed, {c.get('revised', 0)} revised, "
          f"{c.get('judged', 0)} judged, {c.get('fact_checked', 0)} fact-checked.")
    s2 = "Top 3: " + ", ".join(f"{ideas[i].get('title') or i} {ideas[i].get('total')}" for i in top6[:3]) + "."
    return f"{s1} {s2}"


def lessons(run: dict, ideas: dict[str, dict]) -> list[str]:
    out = []
    started = parse_ts(run.get("started_at"))
    kill_reasons = Counter()
    for d in ideas.values():
        if d.get("stage") == "killed" and started and (t := parse_ts(d.get("updated_at"))) and t >= started:
            note = (d.get("stage_note") or "").lower()
            key = ("logistics" if note.startswith("logistics") else "revision cap" if "after 2 revisions" in note
                   else "famous original" if "famous" in note or "clone" in note else "critic")
            kill_reasons[key] += 1
    if kill_reasons:
        out.append("Kills this run by reason: " + ", ".join(f"{k} {v}" for k, v in kill_reasons.most_common()) + ".")
    u = (run.get("usage") or {})
    if u:
        by = u.get("by_node") or {}
        heavy = sorted(by.items(), key=lambda kv: -kv[1].get("tokens", 0))[:2]
        if heavy:
            out.append("Heaviest nodes: " + ", ".join(f"{k} {v.get('tokens', 0):,} tokens/{v.get('searches', 0)} searches" for k, v in heavy) + ".")
    if run.get("stopped_reason"):
        out.append(f"Run cut short: {run['stopped_reason']}.")
    return out[:3]


def notify(text: str, now: datetime) -> str:
    """Push the digest to ntfy.sh if IDEA_LAB_NTFY_TOPIC is set. Quiet hours 21:00–07:00 UK."""
    topic = os.environ.get("IDEA_LAB_NTFY_TOPIC")
    if not topic:
        return "not sent (IDEA_LAB_NTFY_TOPIC not set); see out/digest.txt"
    h = now.astimezone(LONDON).hour
    if h >= 21 or h < 7:
        return "not sent (quiet hours)"
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=text.encode(), method="POST",
                                 headers={"Title": "Idea Lab digest"})
    try:
        urllib.request.urlopen(req, timeout=10).read()
        return "sent to ntfy"
    except Exception as e:  # noqa: BLE001
        return f"ntfy failed: {e}"


__all__ = ["angle_yield", "digest", "summary", "lessons", "notify", "live_sites_line"]
