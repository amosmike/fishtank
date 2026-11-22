"""Apply Michael's decisions. Two inputs, both applied once each, at the start of every run:

1. data/decisions.yaml: hand-written entries (see the file for the format).
2. data/inbox/*.json: change files downloaded from the viewer (out/index.html). Each file is moved
   to data/inbox/applied/ once applied.

This is the only code path allowed to change an idea in a decided stage, because these are his
decisions. Entries already applied are remembered in data/decisions.applied.json by content hash,
so editing or re-saving decisions.yaml never re-applies an old decision.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import config
from .rules import PROTECTED_STAGES
from .store import Store, now_iso, slugify

STATUSES = {"selected", "parked", "rejected", "building", "hosted", "undo"}


def _hash(entry: dict) -> str:
    return hashlib.sha256(json.dumps(entry, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _load_applied(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def decide(store: Store, idea_id: str, status: str, reasons: list[str] | None = None, text: str = "",
           hosted_url: str = "", at: str | None = None) -> str:
    """Apply one decision. Returns a one-line description."""
    if status not in STATUSES:
        raise ValueError(f"{idea_id}: unknown status {status!r} (use one of {sorted(STATUSES)})")
    idea = store.idea(idea_id)
    t = at or now_iso()
    reasons = list(reasons or [])
    note = " · ".join([*reasons, text]).strip(" ·")
    if status == "undo":
        new_stage = "shortlisted" if idea.get("total") is not None else "validated"
        store.update_idea(idea_id, {"stage": new_stage, "decision": None}, by="Michael",
                          note="Decision undone", allow_protected=True)
        return f"{idea_id}: decision undone -> {new_stage}"
    patch: dict = {"stage": status}
    if status in ("selected", "parked", "rejected"):
        patch["decision"] = {"status": status, "reasons": reasons, "text": text, "at": t}
    if hosted_url:
        patch["hosted_url"] = hosted_url
    store.update_idea(idea_id, patch, by="Michael", note=note or ("Live at " + hosted_url if hosted_url else ""),
                      allow_protected=True)
    return f"{idea_id}: {status}" + (f" ({note})" if note else "")


def submit_idea(store: Store, title: str, pitch: str, kind: str = "game", notes: str = "",
                idea_id: str | None = None, at: str | None = None) -> str:
    idea_id = idea_id or "mine-" + slugify(title)
    if store.path("ideas", idea_id).exists():
        return f"{idea_id}: already exists, skipped"
    t = at or now_iso()
    store.create_idea(idea_id, {"title": title, "pitch": pitch, "kind": kind, "notes": notes,
                                "source": "michael", "stage": "submitted", "created_at": t},
                      by="Michael", note="Added (decisions)")
    return f"{idea_id}: submitted"


def update_site(store: Store, site_id: str, patch: dict, create: bool = False) -> str:
    site = store.get("sites", site_id)
    if site is None and not create:
        raise KeyError(f"sites/{site_id} doesn't exist")
    site = {**(site or {}), **patch, "updated_at": now_iso()}
    store.put("sites", site_id, site)
    return f"sites/{site_id}: updated ({', '.join(sorted(patch))})"


# ---- viewer change files -------------------------------------------------------------------------
def apply_op(store: Store, op: dict) -> str | None:
    """Translate one write the viewer recorded into a semantic decision."""
    path, data, kind = op.get("path", ""), op.get("data") or {}, op.get("op")
    coll, _, doc_id = path.partition("/")
    if coll == "ideas" and kind == "set" and data.get("source") == "michael":
        return submit_idea(store, data.get("title", ""), data.get("pitch", ""), data.get("kind", "game"),
                           data.get("notes", ""), idea_id=doc_id, at=data.get("created_at"))
    if coll == "ideas" and kind == "update":
        last = (data.get("history") or [{}])[-1]
        if "decision" in data and data["decision"] is None:
            return decide(store, doc_id, "undo", at=data.get("updated_at"))
        dec = data.get("decision") or {}
        status = dec.get("status") or data.get("stage")
        if status in PROTECTED_STAGES:
            return decide(store, doc_id, status, dec.get("reasons"), dec.get("text") or (last.get("note") or ""),
                          hosted_url=data.get("hosted_url", ""), at=dec.get("at") or data.get("updated_at"))
        return f"{path}: ignored (not a decision)"
    if coll == "sites":
        patch = {k: v for k, v in data.items() if k != "updated_at"}
        return update_site(store, doc_id, patch, create=(kind == "set"))
    if coll == "config":
        return f"{path}: ignored (edit data/config/{doc_id}.json in the repo instead)"
    return f"{path}: ignored"


def apply_all(store: Store, decisions_file: Path | None = None, inbox: Path | None = None) -> list[str]:
    decisions_file = Path(decisions_file or store.root / config.DECISIONS.name)
    inbox = Path(inbox or store.root / "inbox")
    applied_path = store.root / "decisions.applied.json"
    applied = _load_applied(applied_path)
    log: list[str] = []

    if decisions_file.exists():
        import yaml
        doc = yaml.safe_load(decisions_file.read_text()) or {}
        for e in doc.get("decisions") or []:
            h = _hash(e)
            if h in applied:
                continue
            log.append(decide(store, e["id"], e["status"], e.get("reasons"), e.get("text", ""), e.get("hosted_url", "")))
            applied[h] = {"at": now_iso(), "entry": e}
        for e in doc.get("new_ideas") or []:
            h = _hash(e)
            if h in applied:
                continue
            log.append(submit_idea(store, e["title"], e["pitch"], e.get("kind", "game"), e.get("notes", ""), e.get("id")))
            applied[h] = {"at": now_iso(), "entry": e}
        for e in doc.get("site_updates") or []:
            h = _hash(e)
            if h in applied:
                continue
            log.append(update_site(store, e["id"], e.get("patch") or {}, e.get("create", False)))
            applied[h] = {"at": now_iso(), "entry": e}

    if inbox.exists():
        done = inbox / "applied"
        for f in sorted(inbox.glob("*.json")):
            ops = json.loads(f.read_text()).get("ops") or []
            for op in ops:
                msg = apply_op(store, op)
                if msg:
                    log.append(msg)
            done.mkdir(parents=True, exist_ok=True)
            f.replace(done / f.name)

    if log:
        applied_path.write_text(json.dumps(applied, indent=1, ensure_ascii=False) + "\n")
    return log
