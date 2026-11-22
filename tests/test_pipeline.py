"""Queues, decisions, dedupe, stream parsing, routing, evals, and end-to-end runs with a fake model
(no network, no tokens). Uses the fictional sample_data/."""
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from idea_lab import config, decisions, evals, graph, llm, nodes, rules  # noqa: E402
from idea_lab.steps import dedupe, memory  # noqa: E402
from idea_lab.store import Store  # noqa: E402

SAMPLE = ROOT / "sample_data"
NOW = datetime(2026, 11, 16, 9, 0, tzinfo=timezone.utc)
VALIDATED = ["chalk-figures", "ferry-finder"]
RUN = "run-20261108-0849"


def temp_store() -> tuple[Path, Store]:
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(SAMPLE, tmp / "data", ignore=shutil.ignore_patterns("cache", "inbox", "evals"))
    return tmp, Store(tmp / "data")


class Queues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ideas = Store(SAMPLE).ideas()
        cls.q = memory.queues(cls.ideas, NOW)

    def test_validated_waiting_for_the_judge(self):
        self.assertEqual(self.q["validated"], VALIDATED)

    def test_revision_due_when_verdict_is_newer_than_revision(self):
        self.assertEqual(self.q["reviser"], ["river-run"])

    def test_critic2_due_when_revision_is_newer_than_verdict(self):
        self.assertEqual(self.q["critic2"], ["recipe-scaler"])

    def test_owners_new_idea_goes_to_critic1(self):
        self.assertEqual(self.q["critic1"], ["mine-spreadsheet-golf"])

    def test_parked_needing_a_better_angle_is_revived(self):
        self.assertEqual(self.q["revive_parked"], ["charity-shop-map"])

    def test_judge_scope_validated(self):
        self.assertEqual(nodes.judge_targets(self.ideas, "validated"), VALIDATED)


class Decisions(unittest.TestCase):
    def setUp(self):
        self.tmp, self.store = temp_store()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_yaml_applies_once(self):
        f = self.tmp / "d.yaml"
        f.write_text("decisions:\n  - id: tide-ready\n    status: parked\n    reasons: [Good, not now]\n    text: later\n"
                     "new_ideas:\n  - title: Test Idea\n    pitch: a pitch\n")
        log = decisions.apply_all(self.store, f, self.tmp / "inbox")
        self.assertEqual(len(log), 2)
        d = self.store.idea("tide-ready")
        self.assertEqual((d["stage"], d["decision"]["status"], d["decision"]["text"]), ("parked", "parked", "later"))
        self.assertEqual(self.store.idea("mine-test-idea")["stage"], "submitted")
        # undo by hand, then re-apply: the old yaml entry must not come back
        decisions.decide(self.store, "tide-ready", "undo")
        self.assertEqual(decisions.apply_all(self.store, f, self.tmp / "inbox"), [])
        self.assertEqual(self.store.idea("tide-ready")["stage"], "shortlisted")

    def test_viewer_change_file(self):
        inbox = self.tmp / "inbox"
        inbox.mkdir()
        ops = [
            {"op": "update", "path": "ideas/sow-what", "data": {"stage": "selected", "decision": {
                "status": "selected", "reasons": ["Fun to build"], "text": "go", "at": "2026-11-16T10:00:00Z"}}},
            {"op": "set", "path": "ideas/mine-x", "data": {"title": "X", "pitch": "p", "source": "michael", "kind": "tool"}},
            {"op": "update", "path": "sites/dawn-chorus", "data": {"weeks": [{"week_start": "2026-11-09", "visits": 40}]}},
            {"op": "update", "path": "ideas/kettle-cost", "data": {"decision": None, "stage": "shortlisted"}},
            {"op": "set", "path": "config/playbook", "data": {"text": "x"}},
        ]
        (inbox / "c.json").write_text(json.dumps({"ops": ops}))
        log = decisions.apply_all(self.store, self.tmp / "none.yaml", inbox)
        self.assertEqual(self.store.idea("sow-what")["stage"], "selected")
        self.assertEqual(self.store.idea("mine-x")["source"], "michael")
        self.assertEqual(self.store.get("sites", "dawn-chorus")["weeks"][0]["visits"], 40)
        self.assertEqual(self.store.idea("kettle-cost")["stage"], "shortlisted")
        self.assertIn("ignored", log[-1])  # config edits never come from the viewer
        self.assertTrue((inbox / "applied" / "c.json").exists())


class Dedupe(unittest.TestCase):
    def test_code_drops_near_titles(self):
        existing = {"tide-ready": {"title": "Tide Ready"}, "sow-what": {"title": "Sow What?"}}
        kept, dropped = dedupe.code_dedupe([{"title": "The Tide Ready"}, {"title": "Brand New"}, {"title": "Brand new!"}], existing)
        self.assertEqual([k["title"] for k in kept], ["Brand New"])
        self.assertEqual(len(dropped), 2)

    def test_cap(self):
        ideas = [{"title": str(i), "lane": "big" if i < 7 else "week"} for i in range(20)]
        kept, over = dedupe.cap_batch(ideas, 12, 5)
        self.assertEqual(len(kept), 12)
        self.assertEqual(sum(k["lane"] == "big" for k in kept), 5)


class StreamParse(unittest.TestCase):
    def test_counts_searches_and_reads_structured_output(self):
        lines = [
            {"type": "system"},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "WebSearch"}, {"type": "text"}]}},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "WebSearch"},
                                                          {"type": "tool_use", "name": "WebFetch"}]}},
            {"type": "result", "is_error": False, "structured_output": {"a": 1}, "total_cost_usd": 0.1,
             "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100}},
        ]
        data, u = llm.parse_stream("\n".join(json.dumps(x) for x in lines))
        self.assertEqual(data, {"a": 1})
        self.assertEqual((u.searches, u.tokens, u.budget_tokens), (2, 115, 25))

    def test_limit_error(self):
        out = json.dumps({"type": "result", "is_error": True, "result": "usage limit reached|123"})
        with self.assertRaises(llm.LimitError):
            llm.parse_stream(out)


class Routing(unittest.TestCase):
    def test_budget_hit_skips_to_judge(self):
        b = llm.Budget(max_tokens=100, max_searches=10, max_usd=0)
        ctx = nodes.Ctx(Store(Path(tempfile.mkdtemp())), b, "r")
        state = {"plan": graph.make_plan("discovery")}
        self.assertEqual(graph.next_node(state, "scouts", ctx), "dedupe")
        b.record("scouts", llm.Usage(input_tokens=200))
        self.assertEqual(graph.next_node(state, "scouts", ctx), "judge")
        self.assertEqual(graph.next_node(state, "judge", ctx), "close")

    def test_only_plan(self):
        self.assertEqual(graph.make_plan("discovery", ["judge"]), ["memory", "judge", "close"])
        self.assertNotIn("scouts", graph.make_plan("maintenance"))


BLIND = ("source", "notes", "rev_count", "revisions", "history", "decision", "stage", "scores", "total", "judge_notes")


def fake_call(role, payload, schema, *, node, budget, **kw):
    budget.record(node, llm.Usage(input_tokens=1000, output_tokens=100, calls=1))
    if role == "classify":
        return {"ideas": [{"id": i["id"], "category": "test", "mechanic": "test"} for i in payload["ideas"]]}
    if role == "judge":
        for e in payload["to_score"]:
            for k in BLIND:
                assert k not in e, f"judge saw {k}"
        out = []
        for e in payload["to_score"]:
            s = {"demand": 4, "audience": 4, "gap": 4, "build": 4, "share": 5, "ads": 3, "fun": 4}
            fam = "Wordle" if e["id"] == "chalk-figures" else "none"
            out.append({"id": e["id"], "scores": s, "famous_original": fam, "same_theme_version_exists": False,
                        "wedge_is_new_audience_or_mechanic": False, "judge_notes": "fake"})
        return {"ideas": out, "calibration_notes": "too early"}
    if role == "factcheck":
        return {"ideas": [{"id": e["id"], "status": "flagged", "summary": "s", "checks": [],
                           "score_changes": {"demand": +1, "gap": -1}, "missed_competitor_delivers_wedge": False}
                          for e in payload["ideas"]]}
    raise AssertionError(f"unexpected role {role}")


class EndToEndJudge(unittest.TestCase):
    def test_judge_and_factcheck(self):
        tmp, store = temp_store()
        try:
            with mock.patch.object(llm, "call", fake_call), mock.patch.object(config, "OUT", tmp / "out"), \
                    mock.patch.dict("os.environ", {"IDEA_LAB_NTFY_TOPIC": ""}):
                before = store.ideas()
                decided_before = {i: d for i, d in before.items() if rules.is_protected(d)}
                ctx = nodes.Ctx(store, llm.Budget(), RUN)
                state = {"run_id": RUN, "run_type": "discovery", "plan": graph.make_plan("discovery", ["judge", "factcheck"]),
                         "judge_scope": "validated", "done": []}
                graph.run(ctx, state, engine="plain")
            after = store.ideas()
            for i in VALIDATED:
                self.assertIn(after[i]["stage"], ("judged", "shortlisted"), i)
                self.assertEqual(after[i]["total"], rules.total(after[i]["scores"]))
            # famous-original caps applied in code: gap 4->2, share 5->4 (the fact-check may lower gap further)
            cf = after["chalk-figures"]
            self.assertIn("gap 4->2", cf["judge_notes"])
            self.assertLessEqual(cf["scores"]["gap"], 2)
            self.assertEqual(cf["scores"]["share"], 4)
            # fact-check only lowered scores
            for i, d in after.items():
                if (d.get("fact_check") or {}).get("run_id") == RUN:
                    self.assertEqual(d["scores"]["demand"], 4)
                    self.assertEqual(d["scores"]["gap"], max(0, (before[i].get("scores") or {}).get("gap", 4) - 1)
                                     if i not in VALIDATED else d["scores"]["gap"])
            # owner's decisions untouched
            for i, d in decided_before.items():
                self.assertEqual((after[i]["stage"], after[i].get("total")), (d["stage"], d.get("total")), i)
            self.assertEqual(sum(1 for d in after.values() if d.get("stage") == "shortlisted"), 6)
            run = store.run(RUN)
            self.assertEqual(run["status"], "done")
            self.assertGreater(run["usage"]["total"]["calls"], 0)
        finally:
            shutil.rmtree(tmp)


class Resilience(unittest.TestCase):
    def _state(self):
        return {"run_id": RUN, "run_type": "discovery", "plan": graph.make_plan("discovery", ["judge", "factcheck"]),
                "judge_scope": "validated", "done": []}

    def test_resume_after_usage_limit_does_not_redo_finished_nodes(self):
        for engine in ("langgraph", "plain"):
            tmp, store = temp_store()
            calls = []

            def limited(role, *a, **k):
                calls.append(role)
                if role == "factcheck" and calls.count("factcheck") == 1:
                    raise llm.LimitError("usage limit reached (simulated)")
                return fake_call(role, *a, **k)

            try:
                with mock.patch.object(llm, "call", limited), mock.patch.object(config, "OUT", tmp / "out"), \
                        mock.patch.object(config, "STATE", tmp / ".state"):
                    ctx = nodes.Ctx(store, llm.Budget(), RUN)
                    with self.assertRaises(llm.LimitError):
                        graph.run(ctx, self._state(), engine=engine)
                    self.assertEqual(store.run(RUN)["phase"], "factcheck", engine)
                    graph.run(ctx, None, resume=True, engine=engine)
                self.assertEqual(calls.count("judge"), 1, engine)  # the judge was not re-run
                self.assertEqual(calls.count("factcheck"), 2, engine)
                self.assertEqual(store.run(RUN)["status"], "done", engine)
            finally:
                shutil.rmtree(tmp)

    def test_model_error_in_a_node_is_logged_and_the_run_finishes(self):
        tmp, store = temp_store()

        def broken(role, *a, **k):
            if role == "factcheck":
                raise llm.ModelError("bad JSON (simulated)")
            return fake_call(role, *a, **k)

        try:
            with mock.patch.object(llm, "call", broken), mock.patch.object(config, "OUT", tmp / "out"):
                graph.run(nodes.Ctx(store, llm.Budget(), RUN), self._state(), engine="plain")
            run = store.run(RUN)
            self.assertEqual(run["status"], "done")
            self.assertTrue(any(e["agent"] == "error" and "factcheck" in e["msg"] for e in run["log"]))
        finally:
            shutil.rmtree(tmp)


class FullRunWithSimulatedAgents(unittest.TestCase):
    """Every node, end to end, with the demo backend (schema-shaped random answers, no network)."""

    def test_full_discovery_run(self):
        from idea_lab import demo
        for engine in ("langgraph", "plain"):
            tmp, store = temp_store()
            try:
                with mock.patch.object(config, "BACKEND", "demo"), mock.patch.object(demo, "DELAY", 0), \
                        mock.patch.object(config, "OUT", tmp / "out"), mock.patch.object(config, "STATE", tmp / ".state"):
                    before = store.ideas()
                    ctx = nodes.Ctx(store, llm.Budget(), "demo-test")
                    store.put("runs", "demo-test", {"started_at": "2026-11-16T09:00:00Z", "status": "running", "log": []})
                    graph.run(ctx, {"run_id": "demo-test", "run_type": "discovery", "plan": graph.make_plan("discovery"),
                                    "judge_scope": "changed", "done": []}, engine=engine)
                after = store.ideas()
                run = store.run("demo-test")
                self.assertEqual(run["status"], "done", engine)
                self.assertEqual({n: v["status"] for n, v in run["nodes"].items()},
                                 {n: "done" for n in graph.ORDER}, engine)
                new = [i for i in after if i not in before]
                self.assertTrue(new, "scouts should have added ideas")
                for i, d in after.items():
                    self.assertIn(d["stage"], rules.STAGES, i)
                    if d.get("scores") and d.get("total") is not None:
                        self.assertEqual(d["total"], rules.total(d["scores"]), i)
                for i, d in before.items():
                    if rules.is_protected(d):
                        if after[i].get("revived_from_parked"):  # the one approved exception: one revival
                            self.assertIn(after[i]["stage"], ("parked", "revising", "validated", "judged", "shortlisted"), i)
                        else:
                            self.assertEqual(after[i]["stage"], d["stage"], i)
                self.assertNotIn("validated", {after[i]["stage"] for i in VALIDATED})  # all judged
                self.assertGreater(run["usage"]["total"]["calls"], 8)
            finally:
                shutil.rmtree(tmp)


class Evals(unittest.TestCase):
    def test_pairwise_accuracy(self):
        labels = {"a": 2, "b": 2, "c": 0, "d": 0, "e": 1}
        self.assertEqual(evals.pairwise_accuracy({"a": 80, "b": 70, "c": 60, "d": 50, "e": 99}, labels), (1.0, 4))
        self.assertEqual(evals.pairwise_accuracy({"a": 50, "b": 50, "c": 50, "d": 50}, labels), (0.5, 4))
        self.assertEqual(evals.pairwise_accuracy({"a": 10}, {"a": 2}), (None, 0))

    def test_spearman(self):
        self.assertEqual(evals.spearman([1, 2, 3, 4], [0, 1, 1, 2]), 0.949)
        self.assertEqual(evals.spearman([4, 3, 2, 1], [1, 2, 3, 4]), -1.0)
        self.assertIsNone(evals.spearman([1, 1, 1], [0, 1, 2]))

    def test_stability(self):
        s = {k: 3 for k in rules.WEIGHTS}
        runs = {"a": [{"scores": s, "total": 60}, {"scores": {**s, "demand": 4}, "total": 65}],
                "b": [{"scores": s, "total": 60}, {"scores": s, "total": 60}]}
        m = evals.stability_metrics(runs)
        self.assertEqual((m["mean_total_range"], m["max_total_range"]), (2.5, 5))
        self.assertEqual(m["criterion_mean_std"]["demand"], 0.25)

    def test_labels_from_decisions(self):
        labels = evals.labelled(Store(SAMPLE).ideas())
        self.assertEqual(labels, {"charity-shop-map": 1, "dawn-chorus": 2, "kettle-cost": 2, "last-post": 0, "quiz-rota": 0})

    def test_no_label_leak_into_taste(self):
        ideas = Store(SAMPLE).ideas()
        for fold in evals.folds(list(evals.labelled(ideas)), 3):
            p = evals.judge_payload(ideas, fold)
            taste = json.dumps(p["taste"])
            for i in fold:
                self.assertNotIn(ideas[i]["title"], taste, f"{i}'s decision leaked into its own fold's taste")

    def test_eval_run_with_fake_judge(self):
        def judge(role, payload, schema, *, node, budget, **kw):
            budget.record(node, llm.Usage(input_tokens=10, calls=1))
            pick = {"dawn-chorus": 5, "kettle-cost": 4, "charity-shop-map": 3, "last-post": 2, "quiz-rota": 1}
            return {"ideas": [{"id": e["id"], "scores": {k: pick[e["id"]] for k in rules.WEIGHTS},
                               "famous_original": "none", "same_theme_version_exists": False,
                               "wedge_is_new_audience_or_mechanic": False, "judge_notes": ""} for e in payload["to_score"]]}
        with mock.patch.object(llm, "call", judge):
            r = evals.run_judge_eval(Store(SAMPLE), repeats=2, k=2)
        t = r["metrics"]["taste"]
        self.assertEqual((t["pairwise_accuracy"], t["pairs"], t["rank_correlation"]), (1.0, 4, 0.949))
        self.assertEqual(r["metrics"]["stability"]["max_total_range"], 0)
        self.assertEqual(r["usage"]["calls"], 4)
        self.assertEqual(len(r["prompt_sha"]), 12)


if __name__ == "__main__":
    unittest.main()
