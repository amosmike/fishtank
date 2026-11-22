"""Judge evals: does the blind judge agree with the owner's decisions, and does it score consistently?

The labels are the decisions already in the data: picked (selected, building, hosted) = 2, parked = 1,
rejected = 0. Every decision made in the viewer adds a label, so the set grows with use.

Metrics:
- taste.pairwise_accuracy: over every (picked, rejected) pair, how often the judge scores the picked idea
  higher (ties count half). 0.5 is chance. Reported per repeat and on the mean score.
- taste.rank_correlation: Spearman correlation between total and the 0/1/2 label, parked included.
- stability: for each idea, the range of its total across repeats with identical inputs, and the mean
  standard deviation per criterion. This is the drift the pipeline's stability rule exists to contain.
- baseline: the same taste metrics on the totals already stored in the data (the scores the pipeline gave
  at the time), so a prompt or model change can be compared with what's in production.

Leakage control: the judge's normal brief includes a taste summary built from the owner's decisions,
which would hand it the answer. Ideas are split into k folds, and each fold is judged with a taste
summary built only from the decisions outside it. Each judge call also gets the pipeline's blind
view of an idea (no source, notes, stage, decision, scores or revision count).

Scores go through the same post-processing as the pipeline (famous-original caps), so the eval measures
what the leaderboard would show.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from . import config, llm, schemas
from .nodes import evidence
from .rules import WEIGHTS, apply_famous_caps, total
from .steps.memory import taste_summary
from .store import Store

PICKED = {"selected", "building", "hosted"}


# ---- labels -------------------------------------------------------------------------------------
def label(idea: dict) -> int | None:
    status = (idea.get("decision") or {}).get("status")
    stage = idea.get("stage")
    if stage in PICKED or status == "selected":
        return 2
    if status == "parked" or stage == "parked":
        return 1
    if status == "rejected" or stage == "rejected":
        return 0
    return None


def labelled(ideas: dict[str, dict]) -> dict[str, int]:
    """Decided ideas that carry enough evidence to be judged."""
    out = {}
    for i, d in ideas.items():
        lab = label(d)
        if lab is not None and d.get("pitch") and (d.get("wedge") or d.get("competitors") or d.get("critique")):
            out[i] = lab
    return dict(sorted(out.items()))


# ---- metrics (pure) ------------------------------------------------------------------------------
def pairwise_accuracy(scores: dict[str, float], labels: dict[str, int]) -> tuple[float | None, int]:
    pos = [scores[i] for i, lab in labels.items() if lab == 2 and i in scores]
    neg = [scores[i] for i, lab in labels.items() if lab == 0 and i in scores]
    pairs = [(p, n) for p in pos for n in neg]
    if not pairs:
        return None, 0
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in pairs)
    return round(wins / len(pairs), 3), len(pairs)


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda k: xs[k])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if vx == 0 or vy == 0 else round(cov / (vx * vy), 3)


def taste_metrics(scores: dict[str, float], labels: dict[str, int]) -> dict:
    ids = [i for i in labels if i in scores]
    acc, pairs = pairwise_accuracy(scores, labels)
    return {"pairwise_accuracy": acc, "pairs": pairs,
            "rank_correlation": spearman([scores[i] for i in ids], [labels[i] for i in ids])}


def stability_metrics(runs: dict[str, list[dict]]) -> dict:
    """runs: idea id -> list of {"scores", "total"} from repeated identical calls."""
    ranges = {i: max(r["total"] for r in rs) - min(r["total"] for r in rs) for i, rs in runs.items() if len(rs) > 1}
    crit = {k: round(statistics.fmean(statistics.pstdev([r["scores"][k] for r in rs]) for rs in runs.values() if len(rs) > 1), 3)
            for k in WEIGHTS} if ranges else {}
    return {"mean_total_range": round(statistics.fmean(ranges.values()), 2) if ranges else None,
            "max_total_range": max(ranges.values()) if ranges else None,
            "criterion_mean_std": crit}


def folds(ids: list[str], k: int) -> list[list[str]]:
    k = max(1, min(k, len(ids)))
    return [ids[f::k] for f in range(k)]


# ---- running -------------------------------------------------------------------------------------
def prompt_sha() -> str:
    text = (config.PROMPTS / "judge.md").read_text() + (config.PROMPTS / "_rules.md").read_text()
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def judge_payload(ideas: dict[str, dict], fold: list[str]) -> dict:
    held = set(fold)
    rest = {i: d for i, d in ideas.items() if i not in held}
    return {
        "to_score": [{"id": i, **evidence(ideas[i])} for i in fold],
        "reference_top6": [],
        "taste": taste_summary(rest),
        "live_results": [],
    }


def plan(store: Store, repeats: int, k: int) -> dict:
    labels = labelled(store.ideas())
    calls = repeats * len(folds(list(labels), k)) if labels else 0
    model = config.ROLE_MODEL["judge"]
    tin, tout = config.EST_TOKENS["judge"]
    pin, pout = config.PRICE_PER_MTOK.get(model, (0, 0))
    return {"ideas": len(labels), "calls": calls, "model": model,
            "est_tokens": calls * (tin + tout), "est_usd": round(calls * (tin / 1e6 * pin + tout / 1e6 * pout), 2)}


def run_judge_eval(store: Store, repeats: int = 3, k: int = 3, model: str | None = None,
                   budget: llm.Budget | None = None) -> dict:
    ideas = store.ideas()
    labels = labelled(ideas)
    if not labels:
        raise SystemExit("No decided ideas with evidence to evaluate against.")
    budget = budget or llm.Budget()
    model = model or config.ROLE_MODEL["judge"]
    runs: dict[str, list[dict]] = {i: [] for i in labels}
    fails = 0
    for _ in range(repeats):
        for fold in folds(list(labels), k):
            try:
                out = llm.call("judge", judge_payload(ideas, fold), schemas.JUDGE, node="eval", budget=budget, model=model)
            except llm.ModelError:
                fails += 1
                continue
            for r in out.get("ideas") or []:
                i = r.get("id")
                if i not in runs:
                    continue
                s, _ = apply_famous_caps(r.get("scores") or {}, r.get("famous_original") or ideas[i].get("famous_original"),
                                         bool(r.get("same_theme_version_exists")), bool(r.get("wedge_is_new_audience_or_mechanic")))
                runs[i].append({"scores": s, "total": total(s)})

    mean = {i: statistics.fmean(r["total"] for r in rs) for i, rs in runs.items() if rs}
    per_repeat = []
    for rep in range(repeats):
        sc = {i: rs[rep]["total"] for i, rs in runs.items() if len(rs) > rep}
        per_repeat.append(pairwise_accuracy(sc, labels)[0])
    stored = {i: float(ideas[i]["total"]) for i in labels if ideas[i].get("total") is not None}
    drift = [abs(mean[i] - stored[i]) for i in mean if i in stored]
    return {
        "suite": "judge", "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model, "backend": config.BACKEND, "prompt_sha": prompt_sha(), "repeats": repeats, "folds": k,
        "n_ideas": len(labels), "labels": {str(v): list(labels.values()).count(v) for v in (2, 1, 0)},
        "failed_calls": fails,
        "metrics": {
            "taste": {**taste_metrics(mean, labels), "per_repeat_pairwise_accuracy": per_repeat},
            "stability": stability_metrics(runs),
            "baseline_stored_scores": taste_metrics(stored, labels),
            "mean_abs_diff_vs_stored": round(statistics.fmean(drift), 2) if drift else None,
        },
        "per_idea": [{"id": i, "label": labels[i], "stored_total": ideas[i].get("total"),
                      "totals": [r["total"] for r in runs[i]]} for i in labels],
        "usage": budget.to_dict()["total"],
    }


def save(store: Store, report: dict) -> Path:
    d = store.root / "evals"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{report['suite']}-{report['at'].replace(':', '').replace('-', '')}.json"
    p.write_text(json.dumps(report, indent=2) + "\n")
    return p


def history(store: Store) -> list[dict]:
    d = store.root / "evals"
    return [json.loads(p.read_text()) for p in sorted(d.glob("judge-*.json"))] if d.exists() else []


def summary_line(r: dict) -> str:
    m = r["metrics"]
    t, s, b = m["taste"], m["stability"], m["baseline_stored_scores"]
    return (f"{r['at']}  {r['model']:<18} prompt {r['prompt_sha']}  n={r['n_ideas']} x{r['repeats']}  "
            f"taste acc {t['pairwise_accuracy']} (stored {b['pairwise_accuracy']})  rho {t['rank_correlation']}  "
            f"range mean {s['mean_total_range']} max {s['max_total_range']}")
