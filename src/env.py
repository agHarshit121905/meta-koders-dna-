"""
CRISPREnv — Full OpenEnv-compliant environment.

Implements the three required methods:
  reset()         → CRISPRObservation
  step(action)    → (CRISPRObservation, CRISPRReward, done: bool, info: dict)
  state()         → dict

Also exposes close() and task listing helpers.
"""
from __future__ import annotations

import copy
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

# Allow running from repo root or src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from models import CRISPRObservation, CRISPRAction, CRISPRReward
from tasks import TASKS, TaskConfig, compute_episode_score, _is_duplicate
from crispr_engine import find_guides, GuideRNA
from ncbi_fetcher import GENE_CATALOG, fetch_gene_info


# ── Reward shaping constants ─────────────────────────────────────────────────
PENALTY_INVALID_INPUT   = -0.30
PENALTY_WRONG_LENGTH    = -0.20
PENALTY_DUPLICATE_GUIDE = -0.15
BONUS_ABOVE_THRESHOLD   = +0.10   # extra bonus when guide clears task threshold
BONUS_PERFECT_GUIDE     = +0.20   # extra bonus for score >= 0.90


# ── Internal helpers ─────────────────────────────────────────────────────────

def _score_guide(guide_seq: str, all_guides: List[GuideRNA]) -> Optional[GuideRNA]:
    """Return the matching GuideRNA object if the sequence is found in this locus."""
    guide_seq = guide_seq.upper()
    for g in all_guides:
        if g.sequence == guide_seq:
            return g
    return None


def _make_guide_dict(rank: int, guide: GuideRNA, guide_seq: str) -> dict:
    d = guide.to_dict()
    d["rank"] = rank
    d["guide_sequence"] = guide_seq
    return d


def _compute_step_reward(
    total_score: float,
    flags: List[str],
    penalty: float,
    task_threshold: float,
    best_so_far: float,
) -> float:
    """
    Shaped reward at every step. Always returns a value in [0, 1].

    Signal design:
      - Base = total_score (0-1)
      - Bonus for beating the task threshold
      - Bonus for near-perfect guide
      - Bonus for improvement over best so far
      - Deduct flag penalties and input penalties
    """
    reward = total_score

    if total_score >= 0.90:
        reward += BONUS_PERFECT_GUIDE
    elif total_score >= task_threshold:
        reward += BONUS_ABOVE_THRESHOLD

    # Incremental improvement bonus
    if total_score > best_so_far:
        reward += 0.05 * (total_score - best_so_far)

    # Flag deductions
    flag_penalty = 0.0
    for f in flags:
        if "PolIII" in f or "TTTT" in f:
            flag_penalty += 0.08
        elif "GC" in f:
            flag_penalty += 0.04
        elif "folding" in f or "homopolymer" in f:
            flag_penalty += 0.04
    reward -= flag_penalty

    # Input quality penalties
    reward += penalty  # penalty is already ≤ 0

    return round(max(0.0, min(1.0, reward)), 4)


# ── Main environment class ───────────────────────────────────────────────────

class CRISPREnv:
    """
    OpenEnv-compliant CRISPR Guide RNA Design environment.

    Usage
    -----
    env = CRISPREnv(task_name="single_guide_easy")
    obs = env.reset()
    for _ in range(obs.max_steps):
        action = CRISPRAction(guide_sequence="ACGTACGTACGTACGTACGT")
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    """

    TASK_NAMES = list(TASKS.keys())

    def __init__(self, task_name: str = "single_guide_easy"):
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {self.TASK_NAMES}"
            )
        self._task_name = task_name
        self._task: TaskConfig = TASKS[task_name]

        # Episode state
        self._step_number: int = 0
        self._done: bool = False
        self._all_guides: List[GuideRNA] = []         # candidates in current locus
        self._proposed_seqs: List[str] = []           # sequences proposed this episode
        self._episode_history: Dict[str, List[dict]] = {}
        self._best_score_so_far: float = 0.0
        self._current_gene_idx: int = 0               # for multi-gene task
        self._sequence_data: Optional[dict] = None
        self._observation: Optional[CRISPRObservation] = None

    # ── Public API ────────────────────────────────────────────────────────

    def reset(self) -> CRISPRObservation:
        """Start a fresh episode. Returns the initial observation."""
        self._step_number = 0
        self._done = False
        self._proposed_seqs = []
        self._best_score_so_far = 0.0
        self._current_gene_idx = 0

        # Initialise history buckets per task
        if self._task_name == "multi_gene_hard":
            self._episode_history = {"tp53": [], "brca1": [], "ace2": []}
        else:
            self._episode_history = {"guides": []}

        self._load_locus()
        self._observation = self._build_observation()
        return copy.deepcopy(self._observation)

    def step(
        self, action: CRISPRAction
    ) -> Tuple[CRISPRObservation, CRISPRReward, bool, dict]:
        """
        Process one agent action.

        Parameters
        ----------
        action : CRISPRAction  (or dict — auto-coerced)

        Returns
        -------
        observation : CRISPRObservation
        reward      : CRISPRReward
        done        : bool
        info        : dict  (grader explanation, task metadata)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Auto-coerce dict → CRISPRAction
        if isinstance(action, dict):
            action = CRISPRAction(**action)

        self._step_number += 1
        penalty = 0.0
        info: Dict[str, Any] = {
            "task": self._task_name,
            "step": self._step_number,
            "grader_explanation": "",
        }

        # ── Validate guide sequence ───────────────────────────────────────
        guide_seq = action.guide_sequence.upper().strip()

        # Duplicate check
        is_duplicate = guide_seq in self._proposed_seqs
        if is_duplicate:
            penalty += PENALTY_DUPLICATE_GUIDE
            info["warning"] = f"Duplicate guide proposed: {guide_seq}"

        # Check if guide is actually present in the locus
        matched_guide = _score_guide(guide_seq, self._all_guides)

        if matched_guide is None:
            # Guide not found in locus → synthesize a score from scratch
            # (agent may propose a valid sequence not in this exact region)
            # We still score it with the engine by injecting it
            from crispr_engine import GuideRNA as GR, _calc_gc, _calc_folding_dg
            from crispr_engine import _on_target_score, _specificity_score, _manufacturability, _flag_issues
            gc   = _calc_gc(guide_seq)
            dg   = _calc_folding_dg(guide_seq)
            on_t = _on_target_score(guide_seq)
            spec = _specificity_score(guide_seq)
            mfg  = _manufacturability(guide_seq)
            total = round(0.40*on_t + 0.30*spec + 0.20*mfg + 0.10*min(1.0, gc/0.60), 4)
            flags = _flag_issues(guide_seq, gc, dg)
            matched_guide = GR(
                sequence=guide_seq, pam="NGG", position=-1, strand="?",
                gc_content=gc, folding_dg=dg,
                on_target_score=on_t, specificity_score=spec,
                manufacturability=mfg, total_score=total, flags=flags,
            )
            info["note"] = "Guide sequence not found in locus; scored from sequence properties only."

        g = matched_guide
        guide_dict = _make_guide_dict(self._step_number, g, guide_seq)

        # Track
        self._proposed_seqs.append(guide_seq)
        if g.total_score > self._best_score_so_far:
            self._best_score_so_far = g.total_score

        # Store in appropriate bucket
        self._store_guide(guide_dict)

        # ── Shaped reward ─────────────────────────────────────────────────
        step_reward = _compute_step_reward(
            total_score=g.total_score,
            flags=g.flags,
            penalty=penalty,
            task_threshold=self._task.success_threshold,
            best_so_far=self._best_score_so_far,
        )

        # ── Terminal condition ────────────────────────────────────────────
        max_steps_reached = self._step_number >= self._task.max_steps
        task_succeeded    = self._check_early_success()
        self._done        = max_steps_reached or task_succeeded

        # Multi-gene: switch locus every steps_per_gene steps
        if (
            self._task_name == "multi_gene_hard"
            and not self._done
            and self._step_number % self._task.steps_per_gene == 0
        ):
            self._current_gene_idx += 1
            self._load_locus()

        # ── Episode score (only when done) ────────────────────────────────
        episode_score: Optional[float] = None
        if self._done:
            episode_score, grader_exp = compute_episode_score(
                self._task_name, self._episode_history
            )
            info["grader_explanation"] = grader_exp
            info["episode_score"] = episode_score

        # ── Build reward object ───────────────────────────────────────────
        reward = CRISPRReward(
            step_reward=step_reward,
            total_score=g.total_score,
            on_target=round(g.on_target_score, 4),
            specificity=round(g.specificity_score, 4),
            manufacturability=round(g.manufacturability, 4),
            gc_pct=round(g.gc_content * 100, 1),
            folding_dg=round(g.folding_dg, 2),
            flags=g.flags,
            penalty=round(penalty, 4),
            episode_score=episode_score,
        )

        # ── Build next observation ────────────────────────────────────────
        self._observation = self._build_observation()
        return copy.deepcopy(self._observation), reward, self._done, info

    def state(self) -> dict:
        """Return the full current environment state as a plain dict."""
        return {
            "task_name": self._task_name,
            "step_number": self._step_number,
            "done": self._done,
            "best_score_so_far": self._best_score_so_far,
            "proposed_guide_count": len(self._proposed_seqs),
            "episode_history": copy.deepcopy(self._episode_history),
            "current_gene_idx": self._current_gene_idx,
            "current_accession": self._sequence_data.get("accession") if self._sequence_data else None,
            "locus_guide_count": len(self._all_guides),
        }

    def close(self) -> None:
        """Clean up resources (no-op for this environment)."""
        pass

    # ── Private helpers ───────────────────────────────────────────────────

    def _load_locus(self) -> None:
        """Fetch sequence and pre-compute guide candidates for current gene."""
        accession = self._current_accession()
        data = fetch_gene_info(
            accession,
            region_start=self._task.region_start,
            region_length=self._task.region_length,
        )
        if data is None:
            raise RuntimeError(f"Could not fetch sequence for {accession} from NCBI or offline fallback.")
        self._sequence_data = data
        self._all_guides = find_guides(data["sequence"], pam=self._task.pam)

    def _current_accession(self) -> str:
        if self._task_name == "multi_gene_hard":
            genes = [
                self._task.gene_sequence,
                self._task.gene_sequence_2,
                self._task.gene_sequence_3,
            ]
            idx = min(self._current_gene_idx, len(genes) - 1)
            return genes[idx]
        return self._task.gene_sequence

    def _current_gene_name(self) -> str:
        acc = self._current_accession()
        for name, a in GENE_CATALOG.items():
            if a == acc:
                return name
        return acc

    def _store_guide(self, guide_dict: dict) -> None:
        if self._task_name == "multi_gene_hard":
            buckets = ["tp53", "brca1", "ace2"]
            bucket = buckets[min(self._current_gene_idx, 2)]
            self._episode_history[bucket].append(guide_dict)
        else:
            self._episode_history["guides"].append(guide_dict)

    def _check_early_success(self) -> bool:
        """Allow early termination for easy task only (found great guide)."""
        if self._task_name == "single_guide_easy" and self._best_score_so_far >= 0.85:
            return True
        return False

    def _build_observation(self) -> CRISPRObservation:
        seq_data = self._sequence_data or {}
        all_prev = []
        for v in self._episode_history.values():
            all_prev.extend(v)

        return CRISPRObservation(
            gene_name=self._current_gene_name(),
            accession=seq_data.get("accession", self._current_accession()),
            sequence=seq_data.get("sequence", ""),
            region_start=self._task.region_start,
            region_length=self._task.region_length,
            pam=self._task.pam,
            step_number=self._step_number,
            max_steps=self._task.max_steps,
            task_name=self._task_name,
            task_description=self._task.description,
            previous_guides=all_prev,
            best_score_so_far=self._best_score_so_far,
            done=self._done,
        )
