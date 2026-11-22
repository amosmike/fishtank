"""Dice-roll scout briefs (playbook v9), rolled in code with a real random generator.

Rules enforced here:
- one value from each list, rolled with random.SystemRandom (never chosen);
- no audience+format pair that any of the last 7 runs used (re-roll; counted in `rerolls`);
- no two dice scouts in the same run share an audience or a format (re-roll; not counted).
"""
from __future__ import annotations

import random
import re

AUDIENCES = [
    "UK families with young kids", "teens and students", "teachers", "pensioners and older adults",
    "commuters", "portal gamers (Poki/CrazyGames, global)", "football, rugby or cricket fans",
    "runners, cyclists and walkers", "hobbyists (crafts, gardening, music, collecting)", "pet owners",
    "faith communities", "sole traders and small shop owners", "renters and first-time buyers",
    "group chats and friend groups", "global English-speaking casual players",
    "learners of a skill or language",
]
FORMATS = [
    "daily puzzle", "arcade or physics game", "idle, sim or management game",
    "repeat-use tracker or planner", "lookup or reference", "generator or creative toy",
    "map or data visualisation", "quiz or personality test", "multiplayer-by-link game",
    "calculator or checker", "daily reader or ritual (a little every day)", "live board (refreshes itself)",
]
ENGINES = [
    "free open dataset or API", "procedurally generated content", "hand-curated set",
    "the user's own data (CSV or export)",
    "scheduled LLM job that writes or summarises content (bigger-build lane)",
    "live public feed (weather, transport, sport, prices)",
    "community submissions (bigger-build lane)", "none, pure mechanic",
]
ANGLES = {
    "a": '"is there a site that…" posts (go to Mumsnet, r/SomebodyMakeThis, r/AskUK directly)',
    "b": "UK seasonal moments 3–6 months ahead",
    "c": "free open datasets/APIs that could power a one-page tool",
    "d": "browser and daily-puzzle games",
    "e": "everyday UK annoyances and admin",
    "f": "calculators/converters with bad, ad-stuffed incumbents",
}
NO_REPEAT_RUNS = 7


def key(label: str) -> str:
    """Compare labels loosely: lowercase, text before any '(' ('live public feed (...)' == 'live public feed')."""
    return re.sub(r"\s+", " ", label.split("(")[0]).strip().lower()


def recent_pairs(runs: list[dict], n: int = NO_REPEAT_RUNS) -> set[tuple[str, str]]:
    pairs = set()
    for r in runs[-n:]:
        for d in (r.get("agent_stats") or {}).get("dice") or []:
            if d.get("audience") and d.get("format"):
                pairs.add((key(d["audience"]), key(d["format"])))
    return pairs


def roll(n: int, banned_pairs: set[tuple[str, str]], rng: random.Random | None = None) -> list[dict]:
    rng = rng or random.SystemRandom()
    used_a: set[str] = set()
    used_f: set[str] = set()
    out = []
    for i in range(n):
        rerolls = 0
        for _attempt in range(10_000):
            a, f, e = rng.choice(AUDIENCES), rng.choice(FORMATS), rng.choice(ENGINES)
            if key(a) in used_a or key(f) in used_f:
                continue  # same-run clash: re-roll, doesn't count
            if (key(a), key(f)) in banned_pairs:
                rerolls += 1
                continue
            break
        else:
            raise RuntimeError("no legal dice roll left (every audience+format pair is banned)")
        used_a.add(key(a))
        used_f.add(key(f))
        out.append({"scout": f"dice{i + 1}", "audience": a, "format": f, "engine": e, "rerolls": rerolls})
    return out


def angle_yield(runs: list[dict]) -> dict[str, dict]:
    """Sum agent_stats.angle_yield across runs: {angle: {ideas, survived_critics, shortlisted}}."""
    tot: dict[str, dict] = {}
    for r in runs:
        for angle, y in ((r.get("agent_stats") or {}).get("angle_yield") or {}).items():
            t = tot.setdefault(angle, {"ideas": 0, "survived_critics": 0, "shortlisted": 0})
            for k in t:
                try:
                    t[k] += int(y.get(k) or 0)
                except (TypeError, ValueError, AttributeError):
                    pass
    return tot


def pick_angle(runs: list[dict]) -> str:
    """Best-yield lettered angle not used in the last 3 runs (ties: alphabetical)."""
    recent = set()
    for r in runs[-3:]:
        for a in r.get("angles") or []:
            m = re.match(r"\s*([a-f])\s*[:)]", str(a))
            if m:
                recent.add(m.group(1))
    yields = angle_yield(runs)

    def score(letter: str) -> tuple:
        y = [v for k, v in yields.items() if re.match(rf"\s*{letter}\s*[:)]", k)]
        return (-sum(v["shortlisted"] for v in y), -sum(v["survived_critics"] for v in y), letter)

    options = [a for a in ANGLES if a not in recent] or list(ANGLES)
    return sorted(options, key=score)[0]
