# fishtank: context for Claude

Michael's idea factory for his websites: an agent pipeline that generates, critiques, verifies and ranks ideas for small ad-supported sites, which he builds in the separate `web-sites` repo. It is also a portfolio piece for AI Engineer roles, so code, tests and docs should read as a carefully engineered system. Keep the README in that voice (first person, his project). Don't add notes about how it was written.

- Python package `idea_lab/`, a LangGraph graph with a SQLite checkpointer (`.state/`, gitignored). The README has the architecture and the reasons behind it.
- Data:
  - The real data is a clone of the private repo `amosmike/fishtank-data` in `data/` (gitignored here). It holds the ideas, runs, sites, decisions, eval reports, the v11 playbook (`config/playbook.json`, the reference spec) and the migration brief.
  - `sample_data/` is fictional, with example.com URLs, and is used by the tests, CI and the quick start. Never copy real ideas, notes or decisions into this repo.
  - `IDEA_LAB_DATA` picks the data folder.
- Models run through the Claude CLI (`claude -p`) by default. That includes CI, which uses a `claude setup-token` token in the `CLAUDE_CODE_OAUTH_TOKEN` secret. `--backend api` (needs `ANTHROPIC_API_KEY`) is optional and unused. Always `--dry-run` first, and don't start paid runs or evals unless Michael asks.
- Rules that must hold, each with tests:
  - The pipeline never changes an idea in a decided stage (selected, parked, rejected, building, hosted) except to append history. Only `decisions.py` may override.
  - The judge is blind, and code computes totals and caps.
  - Fact-checks only lower scores.
  - Real URLs only.
- Tests: `.venv/bin/python -m unittest discover -s tests` (sample data, fake model, no network). Add a test for any new rule. The judge eval (`python -m idea_lab eval`) is the check for prompt or model changes to the judge.
