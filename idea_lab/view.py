"""Build out/index.html: the claude.ai Idea Lab page, fed from data/ instead of the artifact database.

The page is reference/idea-lab-page.html unchanged, plus a small shim that answers its
`claude.use("db")` calls from data embedded in the file. Its buttons (select, park, reject, undo,
add an idea, log a site's week) still work: each write is applied on the page and queued in a bar at
the bottom, which downloads a change file. Save that file into data/inbox/ and run
`python -m idea_lab apply` (every run also applies it first). Playbook edits are refused: edit
data/config/playbook.json in the repo instead.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config
from .store import Store

PAGE = config.ROOT / "reference" / "idea-lab-page.html"

SHIM = r"""
<script>
window.__IDEA_LAB__ = __DATA__;
(() => {
  "use strict";
  const D = window.__IDEA_LAB__, KEY = "idea-lab:pending:v1";
  let pending = [];
  try { pending = JSON.parse(localStorage.getItem(KEY) || "[]"); } catch {}
  const fires = new Set();
  const coll = c => D[c] || (D[c] = {});
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(pending)); } catch {} };
  const applyOp = o => {
    const [c, id] = o.path.split("/");
    coll(c)[id] = o.op === "set" ? o.data : { ...(coll(c)[id] || {}), ...o.data };
  };
  // Re-apply unsaved changes from an earlier visit, unless the data already has them.
  pending = pending.filter(o => o.built === D.__built || o.at > D.__built);
  pending.forEach(applyOp);
  const notify = () => { fires.forEach(f => f()); renderBar(); };
  const docsOf = (c, order, lim) => {
    let docs = Object.entries(coll(c)).map(([id, data]) => ({ id, data: () => data }));
    if (order) docs.sort((a, b) => {
      const x = a.data()[order.f] || "", y = b.data()[order.f] || "";
      return (x < y ? -1 : x > y ? 1 : 0) * (order.dir === "desc" ? -1 : 1);
    });
    if (lim) docs = docs.slice(0, lim);
    return { docs, metadata: { fromCache: false } };
  };
  const query = (c, order = null, lim = null) => ({
    orderBy: (f, dir) => query(c, { f, dir }, lim),
    limit: n => query(c, order, n),
    onSnapshot(cb) { const f = () => cb(docsOf(c, order, lim)); fires.add(f); f(); return () => fires.delete(f); },
  });
  const record = (op, path, data) => {
    const [c] = path.split("/");
    if (c === "config") { const e = new Error("Edit data/config in the repo instead"); e.code = "read_only"; throw e; }
    const o = { op, path, data, at: new Date().toISOString(), built: D.__built };
    pending.push(o); save(); applyOp(o); notify();
  };
  const db = {
    collection: c => query(c),
    doc: path => ({
      onSnapshot(cb) {
        const f = () => { const [c, id] = path.split("/"); const d = coll(c)[id]; cb({ exists: !!d, data: () => d }); };
        fires.add(f); f(); return () => fires.delete(f);
      },
      set: async data => record("set", path, data),
      update: async data => record("update", path, data),
    }),
  };
  window.claude = { use: async name => name === "db" ? db : name === "user" ? { can: async () => true } : null };

  let bar;
  function renderBar() {
    if (!bar) {
      bar = document.createElement("div");
      bar.id = "il-pending";
      bar.setAttribute("role", "status");
      bar.style.cssText = "position:fixed;left:16px;right:16px;bottom:16px;z-index:50;display:none;gap:12px;align-items:center;flex-wrap:wrap;" +
        "padding:12px 16px;border-radius:10px;background:var(--surface,#fff);color:var(--ink,#111);border:1px solid var(--accent,#1F4FD8);" +
        "box-shadow:0 6px 24px rgba(0,0,0,.18);font:500 14px/1.4 var(--body,system-ui)";
      document.body.appendChild(bar);
      bar.addEventListener("click", e => {
        const act = e.target.dataset.act;
        if (act === "download") {
          const blob = new Blob([JSON.stringify({ exported_at: new Date().toISOString(), ops: pending }, null, 1)], { type: "application/json" });
          const a = document.createElement("a");
          a.href = URL.createObjectURL(blob);
          a.download = "idea-lab-changes-" + new Date().toISOString().replace(/[-:]/g, "").slice(0, 15) + ".json";
          a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
        }
        if (act === "clear" && confirm("Forget these changes? The page reloads with the data as built.")) {
          pending = []; save(); location.reload();
        }
      });
    }
    bar.style.display = pending.length ? "flex" : "none";
    bar.innerHTML = `<span style="flex:1 1 260px">${pending.length} change${pending.length === 1 ? "" : "s"} not applied yet. ` +
      `Download the change file into <code>fishtank/data/inbox/</code>, then run <code>python -m idea_lab apply</code>.</span>` +
      `<button data-act="download" type="button">Download change file</button><button data-act="clear" type="button">Clear</button>`;
  }
  document.addEventListener("DOMContentLoaded", renderBar);
})();
</script>
"""


def build(store: Store, out_dir: Path) -> Path:
    from .store import now_iso

    data: dict = {"__built": now_iso()}
    data["ideas"] = store.ideas()
    data["runs"] = {rid: {k: v for k, v in r.items() if not k.startswith("_") and k not in ("scout_output", "scout_briefs")}
                    for rid, r in ((r["_id"], r) for r in store.runs())}
    data["sites"] = store.all("sites")
    data["config"] = {k: v for k, v in store.all("config").items() if k in ("playbook", "schedule", "agents")}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("<!--", "<\\!--")
    page = PAGE.read_text()
    i = page.index("<script>")
    html = page[:i] + SHIM.replace("__DATA__", blob) + page[i:]
    html = html.replace("<title>Idea Lab</title>", "<title>Idea Lab (local)</title>", 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(html)
    return path
