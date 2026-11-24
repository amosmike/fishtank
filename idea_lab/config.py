"""Everything Michael is likely to tune lives here: models per role, search caps, budgets, paths."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("IDEA_LAB_ROOT", Path(__file__).resolve().parent.parent))
# The real data lives in a separate private repo cloned into data/ (gitignored here).
# Without it, point IDEA_LAB_DATA at sample_data/ (fictional ideas used by the tests and the quick start).
DATA = Path(os.environ.get("IDEA_LAB_DATA", ROOT / "data")).resolve()
SAMPLE_DATA = ROOT / "sample_data"
CACHE = DATA / "cache"
OUT = ROOT / "out"
STATE = ROOT / ".state"
PROMPTS = Path(__file__).resolve().parent / "prompts"
DECISIONS = DATA / "decisions.yaml"

PLAYBOOK_VERSION = "code-1"  # the prose playbook was v11; data/config/playbook.json is the reference spec

# Which backend runs the models:
#   "cli": headless Claude Code (`claude -p`), on Michael's subscription, like tools/whatsitfor/draft.mjs.
#          CI too, via a long-lived subscription token from `claude setup-token` (CLAUDE_CODE_OAUTH_TOKEN).
#   "demo": simulated agents (idea_lab/demo.py), no network, no tokens.
BACKEND = os.environ.get("IDEA_LAB_BACKEND", "cli")

# Model per role. Cheap model for flags and rewrites, mid model for research, strongest for the judge.
ROLE_MODEL = {
    "trends": "claude-sonnet-5-5",
    "scout": "claude-sonnet-5-5",
    "dedupe": "claude-haiku-5-5",
    "classify": "claude-haiku-5-5",   # idea map: category + mechanic for ideas that lack them
    "critic1": "claude-sonnet-5-5",
    "reviser": "claude-haiku-5-5",
    "critic2": "claude-sonnet-5-5",
    "logistics": "claude-sonnet-5-5",
    "demand": "claude-sonnet-5-5",
    "judge": "claude-opus-5-5",
    "factcheck": "claude-sonnet-5-5",
}
ROLE_EFFORT = {  # output_config.effort / claude --effort
    "dedupe": "low", "classify": "low", "reviser": "medium",
    "trends": "medium", "scout": "medium", "critic1": "medium", "critic2": "medium",
    "logistics": "medium", "demand": "medium", "factcheck": "medium", "judge": "high",
}
# Web searches allowed per call (told in the prompt, bounded by --max-turns, counted from the stream).
ROLE_SEARCHES = {
    "trends": 15, "scout": 12, "dedupe": 0, "classify": 0, "critic1": 15, "reviser": 0,
    "critic2": 12, "logistics": 6, "demand": 10, "judge": 0, "factcheck": 15,
}

# Pipeline sizes.
SCOUTS = int(os.environ.get("IDEA_LAB_SCOUTS", 3))
CRITIC_BATCH = 4          # ideas per critic call
DEMAND_TARGETS = 4
LOGISTICS_TARGETS = 16
LOGISTICS_BATCH = 4
FACTCHECK_TOP = 3
MAX_NEW_IDEAS = 12
MAX_BIG_IDEAS = 5
MAX_PARALLEL = int(os.environ.get("IDEA_LAB_PARALLEL", 3))  # concurrent model calls inside a node

# Freshness.
TRENDS_MAX_AGE_DAYS = 7
FACTCHECK_MAX_AGE_DAYS = 30
LOGISTICS_MAX_AGE_DAYS = 30
DEMAND_HUNT_MAX_AGE_DAYS = 7
RIVAL_CACHE_DAYS = 30

# Run budget. When any is hit the graph skips to judge -> close (the old timebox rule).
# Tokens counted = fresh input + output + cache writes + a tenth of cache reads (see llm.Usage).
# USD: Claude Code's API-equivalent cost (not a bill on a subscription, but a fair proxy for how
# much of the usage limit a run eats).
MAX_TOKENS_PER_RUN = int(os.environ.get("IDEA_LAB_MAX_TOKENS", 800_000))
MAX_SEARCHES_PER_RUN = int(os.environ.get("IDEA_LAB_MAX_SEARCHES", 120))
MAX_USD_PER_RUN = float(os.environ.get("IDEA_LAB_MAX_USD", 8.0))

# Rough prices for --dry-run estimates (USD per million tokens). Check the pricing page before relying on them.
PRICE_PER_MTOK = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
PRICE_PER_SEARCH = 0.01  # web search, USD per search (API-equivalent, for estimates)
# Rough token use per call, for estimates only.
EST_TOKENS = {  # (input, output)
    "trends": (40_000, 4_000), "scout": (45_000, 6_000), "dedupe": (15_000, 1_500),
    "classify": (8_000, 2_000), "critic1": (60_000, 10_000), "reviser": (20_000, 4_000),
    "critic2": (50_000, 8_000), "logistics": (40_000, 8_000), "demand": (40_000, 5_000),
    "judge": (60_000, 6_000), "factcheck": (50_000, 5_000),
}

# Retrieval (idea_lab/retrieval): "bge-small" runs BAAI/bge-small-en-v1.5 locally through fastembed;
# "hash" is a deterministic stand-in used by the tests (no download). If the embedder can't load,
# search falls back to BM25 keywords only. IDEA_LAB_RETRIEVAL=off turns retrieval out of the prompts.
EMBEDDER = os.environ.get("IDEA_LAB_EMBEDDER", "bge-small")
RETRIEVAL = os.environ.get("IDEA_LAB_RETRIEVAL", "on") != "off"
CRITIC_PRECEDENTS = 4      # similar past ideas given to each critic
DEDUPE_NEIGHBOURS = 6      # nearest existing ideas per new candidate for the dedupe check
SCOUT_RELATED = 15         # ideas most related to a scout's brief, with pitches
# Second stage: a local cross-encoder re-orders hybrid search's top RERANK_DEPTH (retrieval/rerank.py).
# "off" keeps plain hybrid; "overlap" is the tests' stand-in. RETRIEVAL_MODE is what the pipeline uses.
RERANKER = os.environ.get("IDEA_LAB_RERANKER", "minilm-l6")
RERANK_DEPTH = 20
RETRIEVAL_MODE = os.environ.get("IDEA_LAB_RETRIEVAL_MODE", "rerank")
# Where the vectors live (retrieval/vectorstore.py): "qdrant" (local, in memory, unless
# IDEA_LAB_QDRANT_URL names a server) or "exact" (NumPy). Qdrant falls back to exact if not installed.
VECTOR_STORE = os.environ.get("IDEA_LAB_VECTOR_STORE", "qdrant")
QDRANT_URL = os.environ.get("IDEA_LAB_QDRANT_URL") or None

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
CALL_TIMEOUT_S = int(os.environ.get("IDEA_LAB_CALL_TIMEOUT", 900))
