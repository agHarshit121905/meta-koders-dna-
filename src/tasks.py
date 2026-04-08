"""
Task definitions and deterministic grader functions.
 
Three tasks with clear difficulty progression:
  - single_guide_easy   : find one good guide for TP53
  - ranked_panel_medium : build a diverse panel of 5 guides for BRCA1
  - multi_gene_hard     : design guides across 3 genes with fleet constraints
 
Each grader returns a float in [0.0, 1.0].
All graders are deterministic given the same episode history.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
 
 
# ── Task registry ────────────────────────────────────────────────────────────
 
@dataclass
class TaskConfig:
    name: str
    difficulty: str          # easy | medium | hard
    max_steps: int
    description: str
    gene_sequence: str       # accession key from GENE_CATALOG
    region_start: int
    region_length: int
    pam: str
    success_threshold: float
    # multi-gene only
    gene_sequence_2: Optional[str] = None
    gene_sequence_3: Optional[str] = None
    steps_per_gene: int = 5
 
 
TASKS: dict[str, TaskConfig] = {
    "single_guide_easy": TaskConfig(
        name="single_guide_easy",
        difficulty="easy",
        max_steps=5,
        description=(
            "Design one high-quality CRISPR guide RNA targeting TP53 (tumour suppressor p53). "
            "The agent receives a 300 bp region of the NM_000546 mRNA and must propose a "
            "20-nucleotide guide sequence (A/C/G/T only, no PAM). "
            "Success = any guide with composite total_score >= 0.65 within 5 steps. "
            "Episode score = max total_score achieved across all steps."
        ),
        gene_sequence="NM_000546",   # TP53
        region_start=200,
        region_length=300,
        pam="NGG",
        success_threshold=0.65,
    ),
 
    "ranked_panel_medium": TaskConfig(
        name="ranked_panel_medium",
        difficulty="medium",
        max_steps=10,
        description=(
            "Build a panel of 5 distinct high-quality CRISPR guides for BRCA1 (NM_007294). "
            "Each guide must achieve total_score >= 0.60. "
            "Diversity rule: no two accepted guides may share 8+ consecutive matching nucleotides. "
            "Duplicate or near-duplicate guides are penalised (-0.15 step reward). "
            "Episode score = mean total_score of the best 5 unique guides found. "
            "If fewer than 5 unique guides are proposed, score is scaled proportionally."
        ),
        gene_sequence="NM_007294",   # BRCA1
        region_start=200,
        region_length=400,
        pam="NGG",
        success_threshold=0.60,
    ),
 
    "multi_gene_hard": TaskConfig(
        name="multi_gene_hard",
        difficulty="hard",
        max_steps=15,
        description=(
            "Design CRISPR guides across three disease genes in sequence: "
            "TP53 (steps 1-5) → BRCA1 (steps 6-10) → ACE2 (steps 11-15). "
            "The active gene rotates every 5 steps — the observation updates automatically. "
            "Target: fleet-average best-guide score >= 0.70 across all three genes. "
            "Guides with quality flags (TTTT motif, GC < 35%, ΔG < -10 kcal/mol) incur "
            "a 0.10 penalty per flag on the episode score. "
            "Episode score = mean of best guide per gene × (1 - penalty_sum)."
        ),
        gene_sequence="NM_000546",    # TP53
        gene_sequence_2="NM_007294",  # BRCA1
        gene_sequence_3="NM_021804",  # ACE2
        region_start=200,
        region_length=300,
        pam="NGG",
        success_threshold=0.70,
        steps_per_gene=5,
    ),
}
 
 
# ── Diversity helper ─────────────────────────────────────────────────────────
 
def _shares_kmer(seq_a: str, seq_b: str, k: int = 8) -> bool:
    """Return True if seq_a and seq_b share any k-mer of length k."""
    kmers_b = {seq_b[i:i+k] for i in range(len(seq_b) - k + 1)}
    for i in range(len(seq_a) - k + 1):
        if seq_a[i:i+k] in kmers_b:
            return True
    return False
 
 
def _is_duplicate(candidate: str, accepted: List[str], k: int = 8) -> bool:
    """Return True if candidate is too similar to any already-accepted guide."""
    return any(_shares_kmer(candidate, acc, k) for acc in accepted)
 
 
# ── Graders ──────────────────────────────────────────────────────────────────
 
def grade_single_guide_easy(history: List[dict]) -> Tuple[float, str]:
    """
    Episode grader for single_guide_easy.
 
    history: list of scored guide dicts from all steps this episode.
    Returns (episode_score 0.0-1.0, explanation string).
    """
    if not history:
        return 0.0, "No guides proposed."
 
    best = max(h["total_score"] for h in history)
    explanation = (
        f"Best total_score across {len(history)} guide(s): {best:.3f}. "
        f"Threshold: 0.65. "
        f"{'SUCCESS' if best >= 0.65 else 'BELOW THRESHOLD'}."
    )
    # Score = raw best score (already 0-1, partial credit for any improvement)
    return round(best, 4), explanation
 
 
def grade_ranked_panel_medium(history: List[dict]) -> Tuple[float, str]:
    """
    Episode grader for ranked_panel_medium.
 
    Selects up to 5 unique (diverse) guides that meet the 0.60 threshold,
    returns their mean score scaled to 5 slots.
    """
    if not history:
        return 0.0, "No guides proposed."
 
    accepted: List[dict] = []
    accepted_seqs: List[str] = []
    rejected_dupes = 0
 
    # Walk history in submission order; greedily accept diverse high-quality guides
    for h in sorted(history, key=lambda x: x["total_score"], reverse=True):
        if len(accepted) >= 5:
            break
        seq = h["guide_sequence"]
        if _is_duplicate(seq, accepted_seqs):
            rejected_dupes += 1
            continue
        if h["total_score"] >= 0.60:
            accepted.append(h)
            accepted_seqs.append(seq)
 
    n = len(accepted)
    if n == 0:
        return 0.0, f"No guides met 0.60 threshold. Rejected {rejected_dupes} duplicate(s)."
 
    mean_score = sum(h["total_score"] for h in accepted) / n
 
    # Partial-credit formula with floor:
    #   completeness_factor = max(0.40, n/5)
    # This ensures a single perfect guide (n=1) earns at least 40% of its
    # quality score rather than the overly punishing 20% from plain n/5.
    # Full credit (factor = 1.0) is still only awarded when n == 5.
    completeness = n / 5
    completeness_factor = max(0.40, completeness)
    episode_score = round(mean_score * completeness_factor, 4)
 
    explanation = (
        f"Accepted {n}/5 unique guides (mean score {mean_score:.3f}). "
        f"Completeness {completeness:.1%} (factor={completeness_factor:.2f}, floor=0.40). "
        f"Episode score = {mean_score:.3f} × {completeness_factor:.2f} = {episode_score:.4f}. "
        f"Rejected {rejected_dupes} duplicate(s). "
        f"{'SUCCESS' if mean_score >= 0.60 and n >= 3 else 'PARTIAL'}."
    )
    return episode_score, explanation
 
 
def grade_multi_gene_hard(
    history_tp53: List[dict],
    history_brca1: List[dict],
    history_ace2: List[dict],
) -> Tuple[float, str]:
    """
    Episode grader for multi_gene_hard.
 
    Takes per-gene history lists, computes fleet-average of best guide per gene,
    then applies a penalty for quality flags across all proposed guides.
    """
    all_history = history_tp53 + history_brca1 + history_ace2
 
    def best_score(h_list: List[dict]) -> float:
        return max((h["total_score"] for h in h_list), default=0.0)
 
    scores = [
        best_score(history_tp53),
        best_score(history_brca1),
        best_score(history_ace2),
    ]
    fleet_avg = sum(scores) / 3
 
    # Count flag penalties across all guides
    penalty_sum = 0.0
    for h in all_history:
        flags = h.get("flags", [])
        for flag in flags:
            if any(kw in flag for kw in ("TTTT", "PolIII")):
                penalty_sum += 0.10
            elif "GC" in flag:
                penalty_sum += 0.05
            elif "folding" in flag or "ΔG" in flag:
                penalty_sum += 0.05
 
    penalty_sum = min(penalty_sum, 0.40)   # cap total penalty at 0.40
    episode_score = round(fleet_avg * (1.0 - penalty_sum), 4)
    episode_score = max(0.0, min(1.0, episode_score))
 
    explanation = (
        f"TP53 best: {scores[0]:.3f} | BRCA1 best: {scores[1]:.3f} | ACE2 best: {scores[2]:.3f}. "
        f"Fleet average: {fleet_avg:.3f}. "
        f"Flag penalty: {penalty_sum:.2f}. "
        f"Episode score = {fleet_avg:.3f} × {1-penalty_sum:.2f} = {episode_score:.4f}. "
        f"{'SUCCESS' if fleet_avg >= 0.70 else 'BELOW TARGET 0.70'}."
    )
    return episode_score, explanation
 
 
# ── Convenience dispatch ──────────────────────────────────────────────────────
 
def compute_episode_score(task_name: str, episode_history: dict) -> Tuple[float, str]:
    """
    Central grader dispatcher.
    episode_history shape depends on task:
      - easy/medium: {"guides": [...]}
      - hard: {"tp53": [...], "brca1": [...], "ace2": [...]}
    """
    if task_name == "single_guide_easy":
        return grade_single_guide_easy(episode_history.get("guides", []))
    elif task_name == "ranked_panel_medium":
        return grade_ranked_panel_medium(episode_history.get("guides", []))
    elif task_name == "multi_gene_hard":
        return grade_multi_gene_hard(
            episode_history.get("tp53", []),
            episode_history.get("brca1", []),
            episode_history.get("ace2", []),
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")
 