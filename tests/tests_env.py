"""
tests/test_env.py — deterministic unit tests for the CRISPR OpenEnv environment.

Run: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from src.models import CRISPRObservation, CRISPRAction, CRISPRReward
from src.tasks import (
    grade_single_guide_easy,
    grade_ranked_panel_medium,
    grade_multi_gene_hard,
    _is_duplicate,
    TASKS,
)
from src.env import CRISPREnv


# ── Helper: a minimal scored guide dict ──────────────────────────────────────

def make_guide(seq: str, score: float, flags=None) -> dict:
    return {
        "guide_sequence": seq,
        "sequence": seq,
        "total_score": score,
        "on_target": score,
        "specificity": score,
        "manufacturability": score,
        "gc_pct": 50.0,
        "folding_dg": -5.0,
        "flags": flags or [],
        "rank": 1,
        "position": 0,
        "strand": "+",
        "pam": "NGG",
    }


# ── Model validation tests ────────────────────────────────────────────────────

class TestModels:
    def test_action_valid(self):
        a = CRISPRAction(guide_sequence="ATGATGATGATGATGATGAT")
        assert a.guide_sequence == "ATGATGATGATGATGATGAT"

    def test_action_lowercase_coerced(self):
        a = CRISPRAction(guide_sequence="atgatgatgatgatgatgat")
        assert a.guide_sequence == "ATGATGATGATGATGATGAT"

    def test_action_invalid_bases(self):
        with pytest.raises(Exception):
            CRISPRAction(guide_sequence="ATGATGATGATGATGZZZZ")

    def test_action_wrong_length(self):
        with pytest.raises(Exception):
            CRISPRAction(guide_sequence="ATGATG")

    def test_reward_bounds(self):
        r = CRISPRReward(
            step_reward=0.5, total_score=0.7, on_target=0.8,
            specificity=0.6, manufacturability=0.9,
            gc_pct=50.0, folding_dg=-5.0,
        )
        assert 0.0 <= r.step_reward <= 1.0
        assert 0.0 <= r.total_score <= 1.0


# ── Grader tests ──────────────────────────────────────────────────────────────

class TestGraderEasy:
    def test_empty_history_zero(self):
        score, _ = grade_single_guide_easy([])
        assert score == 0.0

    def test_good_guide_passes(self):
        h = [make_guide("ATGATGATGATGATGATGAT", 0.75)]
        score, exp = grade_single_guide_easy(h)
        assert score >= 0.65
        assert "SUCCESS" in exp

    def test_bad_guide_partial(self):
        h = [make_guide("ATGATGATGATGATGATGAT", 0.40)]
        score, exp = grade_single_guide_easy(h)
        assert score == pytest.approx(0.40, abs=0.01)
        assert "BELOW" in exp

    def test_picks_best_across_steps(self):
        h = [
            make_guide("ATGATGATGATGATGATGAT", 0.50),
            make_guide("GCGCGCGCGCGCGCGCGCGC", 0.80),
            make_guide("TATATATATATATATATATAT"[:20], 0.30),
        ]
        score, _ = grade_single_guide_easy(h)
        assert score == pytest.approx(0.80, abs=0.01)

    def test_score_in_range(self):
        h = [make_guide("ATGATGATGATGATGATGAT", 0.99)]
        score, _ = grade_single_guide_easy(h)
        assert 0.0 <= score <= 1.0


class TestGraderMedium:
    def _five_diverse_guides(self) -> list:
        seqs = [
            "ATGATGATGATGATGATGAT",
            "GCGCGCGCGCGCGCGCGCGC",
            "CCCCATTTGGGCCCAATTTG",
            "TTTACGTATACGTATACGTA",
            "AACCGGTTAACCGGTTAACC",
        ]
        return [make_guide(s, 0.72) for s in seqs]

    def test_five_diverse_high_quality(self):
        h = self._five_diverse_guides()
        score, exp = grade_ranked_panel_medium(h)
        assert score >= 0.60
        assert "5/5" in exp

    def test_empty_history_zero(self):
        score, _ = grade_ranked_panel_medium([])
        assert score == 0.0

    def test_duplicate_guides_rejected(self):
        seq = "ATGATGATGATGATGATGAT"
        h = [make_guide(seq, 0.80), make_guide(seq, 0.80)]
        score, exp = grade_ranked_panel_medium(h)
        # Only 1 unique guide accepted → completeness 1/5 = 0.2
        assert score < 0.80 * 0.5   # well below full credit

    def test_below_threshold_guides_excluded(self):
        h = [make_guide("ATGATGATGATGATGATGAT", 0.45)]
        score, exp = grade_ranked_panel_medium(h)
        assert score == 0.0

    def test_score_in_range(self):
        h = self._five_diverse_guides()
        score, _ = grade_ranked_panel_medium(h)
        assert 0.0 <= score <= 1.0


class TestGraderHard:
    def test_all_perfect_no_flags(self):
        g = make_guide("ATGATGATGATGATGATGAT", 0.80)
        score, exp = grade_multi_gene_hard([g], [g], [g])
        assert score >= 0.70
        assert "SUCCESS" in exp

    def test_empty_all_zero(self):
        score, _ = grade_multi_gene_hard([], [], [])
        assert score == 0.0

    def test_flag_penalty_applied(self):
        bad = make_guide("TTTTTTTTTTTTTTTTTTTT"[:20], 0.80, flags=["PolIII terminator TTTT"])
        good = make_guide("ATGATGATGATGATGATGAT", 0.80)
        score_clean, _ = grade_multi_gene_hard([good], [good], [good])
        score_flagged, _ = grade_multi_gene_hard([bad], [bad], [bad])
        assert score_flagged < score_clean

    def test_score_in_range(self):
        g = make_guide("ATGATGATGATGATGATGAT", 0.80)
        score, _ = grade_multi_gene_hard([g], [g], [g])
        assert 0.0 <= score <= 1.0

    def test_penalty_capped(self):
        # Many flags should not push score below 0
        bad_flags = ["PolIII terminator TTTT", "low GC (<40%)", "strong folding (ΔG=-12.0)"]
        bad = make_guide("TTTTTTTTTTTTTTTTTTTT"[:20], 0.80, flags=bad_flags * 5)
        score, _ = grade_multi_gene_hard([bad], [bad], [bad])
        assert score >= 0.0


# ── Diversity helper tests ────────────────────────────────────────────────────

class TestDiversity:
    def test_identical_is_duplicate(self):
        assert _is_duplicate("ATGATGATGATGATGATGAT", ["ATGATGATGATGATGATGAT"])

    def test_different_not_duplicate(self):
        assert not _is_duplicate("GCGCGCGCGCGCGCGCGCGC", ["ATGATGATGATGATGATGAT"])

    def test_shared_kmer_is_duplicate(self):
        # Share 8-mer ATGATGAT
        assert _is_duplicate("ATGATGATGATGATGATGAT", ["CCCCCCCCATGATGATCCCC"])

    def test_empty_accepted_never_duplicate(self):
        assert not _is_duplicate("ATGATGATGATGATGATGAT", [])


# ── Full environment API tests ─────────────────────────────────────────────────

class TestEnvAPI:
    def test_reset_returns_observation(self):
        env = CRISPREnv("single_guide_easy")
        obs = env.reset()
        assert obs.task_name == "single_guide_easy"
        assert len(obs.sequence) > 0
        assert obs.step_number == 0
        assert obs.done is False
        env.close()

    def test_step_returns_tuple(self):
        env = CRISPREnv("single_guide_easy")
        env.reset()
        obs, reward, done, info = env.step(
            CRISPRAction(guide_sequence="ATGATGATGATGATGATGAT")
        )
        assert isinstance(done, bool)
        assert -1.0 <= reward.step_reward <= 1.0
        assert 0.0 <= reward.total_score <= 1.0
        assert obs.step_number == 1
        env.close()

    def test_step_dict_action(self):
        env = CRISPREnv("single_guide_easy")
        env.reset()
        obs, reward, done, info = env.step({"guide_sequence": "ATGATGATGATGATGATGAT"})
        assert obs.step_number == 1
        env.close()

    def test_state_returns_dict(self):
        env = CRISPREnv("single_guide_easy")
        env.reset()
        s = env.state()
        assert isinstance(s, dict)
        assert "step_number" in s
        assert "best_score_so_far" in s
        env.close()

    def test_episode_ends_at_max_steps(self):
        env = CRISPREnv("single_guide_easy")
        env.reset()
        done = False
        for _ in range(10):   # more than max_steps=5
            if done:
                break
            _, _, done, _ = env.step({"guide_sequence": "ATGATGATGATGATGATGAT"})
        assert done
        env.close()

    def test_step_after_done_raises(self):
        env = CRISPREnv("single_guide_easy")
        env.reset()
        for _ in range(5):
            env.step({"guide_sequence": "ATGATGATGATGATGATGAT"})
        with pytest.raises(RuntimeError):
            env.step({"guide_sequence": "ATGATGATGATGATGATGAT"})
        env.close()

    def test_reset_clears_state(self):
        env = CRISPREnv("single_guide_easy")
        env.reset()
        env.step({"guide_sequence": "ATGATGATGATGATGATGAT"})
        env.reset()
        s = env.state()
        assert s["step_number"] == 0
        assert s["best_score_so_far"] == 0.0
        env.close()

    def test_episode_score_on_done(self):
        env = CRISPREnv("single_guide_easy")
        env.reset()
        final_reward = None
        done = False
        for _ in range(6):
            if done:
                break
            _, reward, done, info = env.step({"guide_sequence": "ATGATGATGATGATGATGAT"})
            if done:
                final_reward = reward
        assert final_reward is not None
        assert final_reward.episode_score is not None
        assert 0.0 <= final_reward.episode_score <= 1.0
        env.close()

    def test_all_three_tasks_run(self):
        for task in CRISPREnv.TASK_NAMES:
            env = CRISPREnv(task)
            obs = env.reset()
            assert obs.task_name == task
            _, reward, _, _ = env.step({"guide_sequence": "ATGATGATGATGATGATGAT"})
            assert -1.0 <= reward.step_reward <= 1.0
            env.close()

    def test_reward_episode_score_in_range(self):
        env = CRISPREnv("ranked_panel_medium")
        env.reset()
        seqs = [
            "ATGATGATGATGATGATGAT",
            "GCGCGCGCGCGCGCGCGCGC",
            "CCCCATTTGGGCCCAATTTG",
            "TTTACGTATACGTATACGTA",
            "AACCGGTTAACCGGTTAACC",
            "GGGTATATATATATATATAT",
            "CCCGATCCCGATCCCGATCC",
            "AAACCCGGGTTTTAAACCCG",
            "TTTTGCGCGCATATATGCGC",
            "ACGTACGTACGTACGTACGT",
        ]
        final_score = None
        for seq in seqs:
            _, reward, done, _ = env.step({"guide_sequence": seq})
            if done and reward.episode_score is not None:
                final_score = reward.episode_score
                break
        assert final_score is None or 0.0 <= final_score <= 1.0
        env.close()


# ── Task config completeness ──────────────────────────────────────────────────

class TestTaskConfig:
    def test_three_tasks_exist(self):
        assert len(TASKS) >= 3

    def test_difficulty_progression(self):
        diffs = [TASKS[t].difficulty for t in TASKS]
        assert "easy" in diffs
        assert "medium" in diffs
        assert "hard" in diffs

    def test_max_steps_positive(self):
        for t in TASKS.values():
            assert t.max_steps > 0

    def test_threshold_in_range(self):
        for t in TASKS.values():
            assert 0.0 < t.success_threshold <= 1.0
