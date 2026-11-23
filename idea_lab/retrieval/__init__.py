"""Retrieval over the lab's own ideas: embeddings, BM25, hybrid search, and the RAG helpers the
pipeline uses (semantic dedupe, lessons from similar ideas for critics, relevant ideas for scouts).
See README "Retrieval" for the design and the evaluation."""
from .index import Index, doc_text, index_for, rrf  # noqa: F401
