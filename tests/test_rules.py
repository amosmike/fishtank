"""Deterministic rules from the playbook. Run: python -m unittest discover -s tests (from idea-lab/)."""
import random
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from idea_lab import dice, rules  # noqa: E402
from idea_lab.store import Store  # noqa: E402

SAMPLE = ROOT / "sample_data"
REAL = ROOT / "data"  # the private data repo, when cloned; tests that use it skip otherwise
DATASETS = [SAMPLE] + ([REAL] if (REAL / "ideas").is_dir() else [])

FULL = {"demand": 5, "audience": 5, "gap": 5, "build": 5, "share": 5, "ads": 5, "fun": 5}


class Total(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(rules.total(FULL), 100)
        self.assertEqual(rules.total({k: 0 for k in FULL}), 0)
        # 3*3 + 2*3*4 + 1*3*2 = 39 -> 39/65*100 = 60
        self.assertEqual(rules.total({k: 3 for k in FULL}), 60)

    def test_rounds_half_up(self):
        # sum 50 -> 76.923 -> 77 ; sum 13 -> 20 exactly
        s = {"demand": 4, "audience": 4, "gap": 4, "build": 4, "share": 4, "ads": 4, "fun": 2}
        self.assertEqual(rules.total(s), round(50 / 65 * 100))

    def test_matches_every_stored_total(self):
        for ds in DATASETS:
            n = 0
            for i, d in Store(ds).ideas().items():
                s = d.get("scores")
                if s and d.get("total") is not None and all(k in s for k in rules.WEIGHTS):
                    self.assertEqual(rules.total(s), d["total"], f"{ds.name}/{i}")
                    n += 1
            self.assertGreater(n, 5, ds)

    def test_clamps_out_of_range(self):
        self.assertEqual(rules.total({**FULL, "demand": 9}), 100)
        self.assertEqual(rules.total({**FULL, "fun": -3}), rules.total({**FULL, "fun": 0}))


class FamousOriginal(unittest.TestCase):
    def test_no_original_no_caps(self):
        s, notes = rules.apply_famous_caps(FULL, "none", True, False)
        self.assertEqual(s, FULL)
        s, _ = rules.apply_famous_caps(FULL, "none (closest is putt.day)", True, False)
        self.assertEqual(s["gap"], 5)

    def test_gap_capped_at_2_and_share_at_4(self):
        s, notes = rules.apply_famous_caps(FULL, "Wordle", False, False)
        self.assertEqual((s["gap"], s["share"]), (2, 4))
        self.assertEqual(len(notes), 2)

    def test_gap_1_when_same_theme_exists_even_with_new_audience(self):
        s, _ = rules.apply_famous_caps(FULL, "NYT Connections", True, True)
        self.assertEqual(s["gap"], 1)

    def test_new_audience_lifts_gap_cap_but_not_share(self):
        s, _ = rules.apply_famous_caps(FULL, "Worldle", False, True)
        self.assertEqual((s["gap"], s["share"]), (5, 4))


class Revisions(unittest.TestCase):
    def test_third_revise_is_a_kill(self):
        stage, note = rules.critic_outcome({"rev_count": 2}, "revise", "still vague", round_no=2)
        self.assertEqual(stage, "killed")
        self.assertIn("after 2 revisions", note)

    def test_second_revise_stays_revising(self):
        self.assertEqual(rules.critic_outcome({"rev_count": 1}, "revise", round_no=2)[0], "revising")

    def test_michaels_ideas_not_killed_on_first_pass(self):
        m = {"source": "michael"}
        self.assertEqual(rules.critic_outcome(m, "kill", "crowded", "incumbent", 1)[0], "revising")
        self.assertEqual(rules.critic_outcome(m, "kill", "trademark", "policy", 1)[0], "killed")
        self.assertEqual(rules.critic_outcome(m, "kill", "crowded", "incumbent", 2)[0], "killed")

    def test_logistics_blocker(self):
        self.assertEqual(rules.logistics_outcome({}, "go", [], False, False), (None, ""))
        self.assertEqual(rules.logistics_outcome({}, "blocker", ["no licence"], False, True)[0], "killed")
        self.assertEqual(rules.logistics_outcome({}, "blocker", ["costly"], False, False)[0], "revising")
        self.assertEqual(rules.logistics_outcome({"rev_count": 2}, "blocker", ["x"], False, False)[0], "killed")
        m = {"source": "michael"}
        self.assertEqual(rules.logistics_outcome(m, "blocker", ["no licence"], False, True)[0], "revising")
        self.assertEqual(rules.logistics_outcome(m, "blocker", ["trademark"], True, True)[0], "killed")


class FactCheck(unittest.TestCase):
    def test_only_lowers_max_two(self):
        s = {k: 3 for k in FULL}
        out, _ = rules.apply_fact_check(s, {"demand": +2, "gap": -5})
        self.assertEqual(out["demand"], 3)
        self.assertEqual(out["gap"], 1)

    def test_never_below_zero(self):
        out, _ = rules.apply_fact_check({**FULL, "gap": 1}, {"gap": -2})
        self.assertEqual(out["gap"], 0)

    def test_ignores_junk(self):
        out, notes = rules.apply_fact_check(FULL, {"vibes": -2, "demand": "x"})
        self.assertEqual(out, FULL)
        self.assertEqual(notes, [])


class Stability(unittest.TestCase):
    def test_moves_at_most_one_without_new_evidence(self):
        prev = {k: 3 for k in FULL}
        new = {**prev, "demand": 5, "gap": 0}
        out, notes = rules.apply_stability(prev, new, new_evidence=False)
        self.assertEqual((out["demand"], out["gap"]), (4, 2))
        self.assertEqual(len(notes), 2)

    def test_free_with_new_evidence(self):
        prev = {k: 3 for k in FULL}
        out, _ = rules.apply_stability(prev, {**prev, "demand": 5}, new_evidence=True)
        self.assertEqual(out["demand"], 5)


class Protected(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = Store(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_decided_stages_only_get_history(self):
        for stage in sorted(rules.PROTECTED_STAGES):
            iid = f"x-{stage}"
            self.store.put("ideas", iid, {"title": "X", "pitch": "p", "stage": stage, "total": 50, "history": []})
            self.store.update_idea(iid, {"stage": "killed", "total": 99, "pitch": "changed"}, by="judge", note="try")
            d = self.store.idea(iid)
            self.assertEqual((d["stage"], d["total"], d["pitch"]), (stage, 50, "p"), stage)
            self.assertEqual(len(d["history"]), 1)
            self.assertIn("not applied", d["history"][0]["note"])

    def test_undecided_stages_update(self):
        self.store.put("ideas", "y", {"title": "Y", "stage": "judged", "history": []})
        self.store.update_idea("y", {"stage": "shortlisted"}, by="judge")
        self.assertEqual(self.store.idea("y")["stage"], "shortlisted")

    def test_revived_parked_idea_is_not_frozen_by_old_decision(self):
        self.assertFalse(rules.is_protected({"stage": "judged", "decision": {"status": "parked"}}))


class Dice(unittest.TestCase):
    def test_no_repeat_pair_within_7_runs(self):
        banned = {(dice.key(a), dice.key(f)) for a in dice.AUDIENCES[:10] for f in dice.FORMATS}
        rng = random.Random(1)
        for _ in range(50):
            for r in dice.roll(3, banned, rng):
                self.assertNotIn((dice.key(r["audience"]), dice.key(r["format"])), banned)

    def test_no_shared_audience_or_format_in_a_run(self):
        rng = random.Random(2)
        for _ in range(200):
            rolls = dice.roll(5, set(), rng)
            self.assertEqual(len({r["audience"] for r in rolls}), 5)
            self.assertEqual(len({r["format"] for r in rolls}), 5)

    def test_recent_pairs_reads_last_7_runs_only(self):
        runs = [{"agent_stats": {"dice": [{"audience": f"a{i}", "format": "f"}]}} for i in range(9)]
        pairs = dice.recent_pairs(runs)
        self.assertNotIn(("a0", "f"), pairs)
        self.assertNotIn(("a1", "f"), pairs)
        self.assertIn(("a2", "f"), pairs)
        self.assertEqual(len(pairs), 7)

    def test_labels_match_stored_runs(self):
        seen = [d for ds in DATASETS for r in Store(ds).runs() for d in (r.get("agent_stats") or {}).get("dice") or []]
        self.assertTrue(seen)
        keys_a = {dice.key(a) for a in dice.AUDIENCES}
        keys_f = {dice.key(f) for f in dice.FORMATS}
        for d in seen:
            self.assertIn(dice.key(d["audience"]), keys_a, d)
            self.assertIn(dice.key(d["format"]), keys_f, d)

    def test_uses_system_random_by_default(self):
        self.assertEqual(len(dice.roll(2, set())), 2)

    def test_impossible_raises(self):
        banned = {(dice.key(a), dice.key(f)) for a in dice.AUDIENCES for f in dice.FORMATS}
        with self.assertRaises(RuntimeError):
            dice.roll(1, banned, random.Random(0))


class Leaderboard(unittest.TestCase):
    def test_top6_excludes_decided_and_unscored(self):
        ideas = Store(SAMPLE).ideas()
        top = rules.top_n(ideas, 6)
        self.assertEqual(top[:2], ["tide-ready", "sow-what"])
        for decided in ("kettle-cost", "dawn-chorus", "last-post", "charity-shop-map"):
            self.assertNotIn(decided, top)
        self.assertNotIn("ferry-finder", top)  # validated, not yet scored

    def test_ties_break_alphabetically(self):
        ideas = {"b": {"stage": "judged", "total": 60}, "a": {"stage": "judged", "total": 60}}
        self.assertEqual(rules.top_n(ideas), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
