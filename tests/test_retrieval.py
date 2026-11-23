"""Retrieval: BM25, fusion, the incremental index, search modes, the eval's labels, and the RAG context
the pipeline sends. Uses the deterministic hash embedder (no model download)."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("IDEA_LAB_EMBEDDER", "hash")

import numpy as np  # noqa: E402

from idea_lab import nodes  # noqa: E402
from idea_lab.retrieval import evaluate, rag, rrf  # noqa: E402
from idea_lab.retrieval.bm25 import BM25, tokens  # noqa: E402
from idea_lab.retrieval.embed import HashEmbedder  # noqa: E402
from idea_lab.retrieval.index import Index, doc_text, index_for  # noqa: E402
from idea_lab.store import Store  # noqa: E402

SAMPLE = ROOT / "sample_data"


class Bm25(unittest.TestCase):
    def test_rare_words_count_more_and_tf_saturates(self):
        docs = [tokens(t) for t in ["daily county outline game", "daily bird song game", "daily game game game game"]]
        bm = BM25(docs)
        s = bm.scores(tokens("county game"))
        self.assertEqual(int(np.argmax(s)), 0)            # "county" is rare, so doc 0 wins
        self.assertLess(s[2], 2 * s[1])                     # four "game"s are worth far less than 4x one

    def test_tokens_drop_stopwords_and_fold_plurals(self):
        self.assertEqual(tokens("The Games of the Counties"), ["game", "countie"])


class Fusion(unittest.TestCase):
    def test_rrf_rewards_agreement_and_ignores_score_scales(self):
        fused = [d for d, _ in rrf([["a", "b", "c"], ["b", "c", "a"]])]
        self.assertEqual(fused[0], "b")                     # ranked 2nd and 1st beats 1st and 3rd
        only_one = dict(rrf([["x"], []]))
        self.assertAlmostEqual(only_one["x"], 1 / 61)


class IndexBehaviour(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ideas = {"a": {"title": "County Outline", "pitch": "Guess the UK county from its shape"},
                      "b": {"title": "Birdsong", "pitch": "Name the garden bird from its song"},
                      "c": {"title": "Kettle Cost", "pitch": "What a boil costs at the price cap"}}

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_only_changed_ideas_are_re_embedded_and_cache_survives(self):
        idx = Index(HashEmbedder(), self.tmp).update(self.ideas)
        self.assertEqual(idx.embedded_last_update, 3)
        self.assertEqual(idx.update(self.ideas).embedded_last_update, 0)
        self.ideas["b"] = {**self.ideas["b"], "pitch": "Name the bird from its dawn song"}
        self.assertEqual(idx.update(self.ideas).embedded_last_update, 1)
        fresh = Index(HashEmbedder(), self.tmp).update(self.ideas)      # a new process reads the disk cache
        self.assertEqual(fresh.embedded_last_update, 0)
        np.testing.assert_allclose(fresh.vecs, idx.vecs)

    def test_modes_exclusions_and_similar(self):
        idx = Index(HashEmbedder(), None).update(self.ideas)
        for mode in ("bm25", "vector", "hybrid"):
            self.assertEqual(idx.search("guess the county shape", 1, mode)[0][0], "a", mode)
        self.assertNotIn("a", [i for i, _ in idx.similar("a", 5)])
        self.assertEqual({i for i, _ in idx.search("county", 5, "vector", allow={"b", "c"})}, {"b", "c"})

    def test_without_an_embedder_search_falls_back_to_bm25(self):
        idx = Index(None, None).update(self.ideas)
        self.assertEqual(idx.search("kettle price", 1, "hybrid")[0][0], "c")

    def test_doc_text_includes_map_tags(self):
        self.assertEqual(doc_text({"title": "T", "pitch": "P", "category": "daily game", "mechanic": "guess"}),
                         "T. P (daily game; guess)")


class EvalLabels(unittest.TestCase):
    def test_resolve_claims(self):
        ideas = {"form-line": {"title": "Form Line"}, "countle": {"title": "Countle: daily county guess"}}
        self.assertEqual(evaluate.resolve("form-line", ideas), "form-line")
        self.assertEqual(evaluate.resolve("Countle", ideas), "countle")
        self.assertEqual(evaluate.resolve("Something (countle)", ideas), "countle")
        self.assertIsNone(evaluate.resolve("none: it's new", ideas))
        self.assertIsNone(evaluate.resolve("Totally different", ideas))

    def test_report_shape_on_sample_data(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            shutil.copytree(SAMPLE, tmp / "d")
            r = evaluate.run(Store(tmp / "d"))
            self.assertEqual(set(r["results"]), {"nearest", "revision"})
            self.assertEqual(set(r["results"]["revision"]), {"bm25", "vector", "hybrid"})
            self.assertGreaterEqual(r["results"]["revision"]["hybrid"]["n"], 1)
            self.assertIn("R@5", evaluate.table(r))
        finally:
            shutil.rmtree(tmp)


class RagContext(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copytree(SAMPLE, self.tmp / "d")
        self.store = Store(self.tmp / "d")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_critics_get_precedents_with_outcomes(self):
        p = nodes._critic_payload(self.store.idea("river-run"), "river-run", store=self.store)
        ids = [x["id"] for x in p["similar_past"]]
        self.assertTrue(ids)
        self.assertNotIn("river-run", ids)
        self.assertTrue(all(x["outcome"] for x in p["similar_past"]))
        self.assertNotIn("similar_past", nodes._critic_payload(self.store.idea("river-run"), "river-run"))

    def test_outcome_lines(self):
        ideas = self.store.ideas()
        self.assertTrue(rag.outcome(ideas["weatherle"]).startswith("killed:"))
        self.assertTrue(rag.outcome(ideas["last-post"]).startswith("rejected by the owner"))
        self.assertIn("/100", rag.outcome(ideas["tide-ready"]))

    def test_index_follows_new_ideas(self):
        idx = index_for(self.store)
        n = len(idx.ids)
        self.store.create_idea("new-one", {"title": "Tide Times Plus", "pitch": "Tides for walkers", "stage": "generated"}, by="t")
        self.assertEqual(len(index_for(self.store).ids), n + 1)
        self.assertEqual(index_for(self.store).embedded_last_update, 0)


if __name__ == "__main__":
    unittest.main()
