"""The retrieval the pipeline uses (RAG over the lab's own history).

- neighbours(): the existing ideas nearest a new candidate, for the dedupe check (instead of sending
  every pitch in the lab to the model).
- precedents(): the past ideas most similar to one a critic is reviewing, with what happened to them
  (killed and why, scores, the owner's decisions in his words). Critics must cite the ids they used.
- related(): the existing ideas most related to a scout's brief, so scouts see the relevant history
  in full while the prompt stays the same size as the lab grows.

Everything degrades gracefully: with retrieval off (IDEA_LAB_RETRIEVAL=off) or numpy missing, the
callers fall back to the pre-retrieval behaviour.
"""
from __future__ import annotations

from .. import config
from ..rules import PROTECTED_STAGES
from ..store import Store


def available() -> bool:
    if not config.RETRIEVAL:
        return False
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def _index(store: Store):
    from .index import index_for
    return index_for(store)


def outcome(d: dict) -> str:
    """One line on what happened to an idea, for the critics' precedents and the scouts' brief."""
    stage = d.get("stage")
    dec = d.get("decision") or {}
    if stage in PROTECTED_STAGES and dec:
        words = dec.get("text") or ", ".join(dec.get("reasons") or [])
        return f"{stage} by the owner" + (f": “{words[:160]}”" if words else "")
    if stage == "killed":
        return f"killed: {(d.get('stage_note') or 'no reason recorded')[:160]}"
    if stage in ("judged", "shortlisted") and d.get("total") is not None:
        return f"scored {d['total']}/100" + (" (top 6)" if stage == "shortlisted" else "")
    return f"in progress ({stage})"


def _brief(i: str, d: dict, pitch_len: int = 140) -> dict:
    return {"id": i, "title": d.get("title") or i, "pitch": (d.get("pitch") or "")[:pitch_len], "outcome": outcome(d)}


def neighbours(store: Store, text: str, k: int | None = None) -> list[dict]:
    idx = _index(store)
    ideas = store.ideas()
    return [_brief(i, ideas[i], 110) | {"score": round(s, 4)} for i, s in idx.search(text, k or config.DEDUPE_NEIGHBOURS, config.RETRIEVAL_MODE) if i in ideas]


def precedents(store: Store, idea_id: str, k: int | None = None) -> list[dict]:
    idx = _index(store)
    ideas = store.ideas()
    return [_brief(i, ideas[i]) for i, _ in idx.similar(idea_id, k or config.CRITIC_PRECEDENTS, config.RETRIEVAL_MODE) if i in ideas]


def related(store: Store, text: str, k: int | None = None) -> list[dict]:
    idx = _index(store)
    ideas = store.ideas()
    return [_brief(i, ideas[i]) for i, _ in idx.search(text, k or config.SCOUT_RELATED, config.RETRIEVAL_MODE) if i in ideas]
