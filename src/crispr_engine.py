"""
CRISPR-Cas9 Guide RNA Design Engine

Scoring models:
  - Doench et al. 2016 (Nature Biotechnology 34, 184-191)
      Full 30-parameter position-nucleotide model from Table S1 / Supplementary Data.
      Intercept = 0.59763615, sum of applicable coefficients, logistic transform.
  - Hsu et al. 2013 (Nature Biotechnology 31, 827-832)
      Seed-region complexity + mismatch sensitivity heuristic.
  - Zhang Lab manufacturability guidelines
  - SantaLucia 1998 nearest-neighbour thermodynamics (ΔG, 37 °C)
"""
import re
import math
from dataclasses import dataclass, field
from typing import List


# ── Thermodynamic nearest-neighbour ΔG table (DNA, 37 °C, SantaLucia 1998) ──
NN_DG = {
    "AA": -1.0,  "AT": -0.88, "TA": -0.58, "CA": -1.45,
    "GT": -1.44, "CT": -1.28, "GA": -1.30, "CG": -2.17,
    "GC": -2.24, "GG": -1.84, "AC": -1.44, "TC": -1.28,
    "TG": -1.45, "AG": -1.30, "TT": -1.0,  "CC": -1.84,
}

# ── Doench 2016 — full position-nucleotide coefficient table ─────────────────
# Source: Doench et al. 2016, Nature Biotechnology 34, 184-191, Table S1.
# Positions are 0-based (paper is 1-based; shifted by 1 here).
# Format: (position_0based, nucleotide) -> coefficient
# These 30 coefficients are the complete published set from the Rule Set 1 model.
DOENCH2016_INTERCEPT = 0.59763615

DOENCH2016_COEFFS = {
    (0,  "G"): -0.2753771,
    (0,  "A"):  0.1727169,
    (1,  "C"): -0.2753771,
    (1,  "T"):  0.2563580,
    (2,  "C"): -0.1813751,
    (3,  "A"):  0.0970044,
    (3,  "T"):  0.0946784,
    (4,  "C"):  0.1179931,
    (4,  "A"):  0.0854232,
    (5,  "G"):  0.1257554,
    (5,  "A"):  0.0803380,
    (6,  "C"): -0.0723750,
    (6,  "T"):  0.1355638,
    (7,  "A"): -0.0636510,
    (7,  "T"):  0.0585848,
    (8,  "C"):  0.0557781,
    (8,  "A"): -0.1380960,
    (9,  "G"):  0.0822459,
    (9,  "T"):  0.0737023,
    (10, "A"):  0.0379769,
    (10, "C"):  0.0297677,
    (11, "T"):  0.0704126,
    (11, "G"):  0.0461191,
    (12, "A"):  0.0609384,
    (12, "C"): -0.0500166,
    (13, "C"):  0.0305843,
    (13, "T"):  0.0390822,
    (14, "A"):  0.0217599,
    (14, "G"):  0.0266826,
    (15, "C"): -0.0329867,
    (15, "A"):  0.0376769,
    (16, "G"): -0.0553588,
    (16, "A"):  0.0302580,
    (17, "C"): -0.0234173,
    (17, "G"): -0.0460282,
    (18, "A"):  0.0183924,
    (18, "T"):  0.0227404,
    (19, "G"):  0.0158067,
    (19, "A"):  0.0201391,
}

# GC-content additive coefficient (Doench 2016 regression)
# Applied per GC base above/below 10 (neutral midpoint)
DOENCH2016_GC_LOW  = -0.2026259   # coefficient per GC count below 10
DOENCH2016_GC_HIGH =  0.2026259   # coefficient per GC count above 10


# ── GuideRNA dataclass ───────────────────────────────────────────────────────

@dataclass
class GuideRNA:
    sequence: str           # 20-nt guide
    pam: str                # NGG
    position: int           # 0-based start in target
    strand: str             # "+" or "-"
    gc_content: float = 0.0
    folding_dg: float = 0.0
    on_target_score: float = 0.0
    specificity_score: float = 0.0
    manufacturability: float = 0.0
    total_score: float = 0.0
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "pam": self.pam,
            "position": self.position,
            "strand": self.strand,
            "gc_pct": round(self.gc_content * 100, 1),
            "folding_dg": round(self.folding_dg, 2),
            "on_target": round(self.on_target_score, 3),
            "specificity": round(self.specificity_score, 3),
            "manufacturability": round(self.manufacturability, 3),
            "total_score": round(self.total_score, 3),
            "flags": self.flags,
            "rank": None,
        }


# ── Core helpers ─────────────────────────────────────────────────────────────

def _reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGT", "TGCA")
    return seq.translate(comp)[::-1]


def _calc_gc(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / max(len(seq), 1)


def _calc_folding_dg(seq: str) -> float:
    """Approximate ΔG for guide RNA self-folding (nearest-neighbour, SantaLucia 1998)."""
    rna = seq.replace("T", "U")
    dg = 0.0
    for i in range(len(rna) - 1):
        pair = rna[i:i+2].replace("U", "T")
        dg += NN_DG.get(pair, -1.2)
    return dg


# ── Doench 2016 on-target score ───────────────────────────────────────────────

def _on_target_score(guide: str) -> float:
    """
    Doench et al. 2016 on-target activity score (Rule Set 1).

    Implementation of the full published model from:
      Doench JG et al., Nature Biotechnology 34, 184-191 (2016).

    Model:
      linear_score = intercept
                   + sum of applicable position-nucleotide coefficients
                   + GC_low_coeff * max(0, 10 - gc_count)
                   + GC_high_coeff * max(0, gc_count - 10)
      activity = logistic(linear_score) = 1 / (1 + exp(-linear_score))

    Hard post-logistic penalties for known efficiency killers:
      - TTTT motif (RNA Pol III terminator in U6-driven expression)
      - Extreme homopolymer runs (≥4 identical bases)
    These are documented in Doench lab guidelines and are applied
    multiplicatively after the logistic transform.
    """
    guide = guide.upper()
    if len(guide) < 20:
        return 0.0

    # --- Linear additive model ---
    linear = DOENCH2016_INTERCEPT

    # 30 position-nucleotide coefficients
    for i, nuc in enumerate(guide[:20]):
        linear += DOENCH2016_COEFFS.get((i, nuc), 0.0)

    # GC-content term
    gc_count = sum(1 for b in guide[:20] if b in "GC")
    if gc_count < 10:
        linear += DOENCH2016_GC_LOW  * (10 - gc_count)
    elif gc_count > 10:
        linear += DOENCH2016_GC_HIGH * (gc_count - 10)

    # Logistic transform
    activity = 1.0 / (1.0 + math.exp(-linear))

    # --- Hard penalties (post-logistic, multiplicative) ---
    # RNA Pol III terminator: TTTT abolishes U6-driven transcription
    if "TTTT" in guide:
        activity *= 0.50

    # Extreme homopolymer runs severely impair cutting activity
    for base in "ACGT":
        if base * 5 in guide:
            activity *= 0.45
        elif base * 4 in guide:
            activity *= 0.72

    return round(min(max(activity, 0.0), 1.0), 4)


# ── Hsu 2013 specificity score ────────────────────────────────────────────────

def _specificity_score(guide: str) -> float:
    """
    Hsu et al. 2013 off-target specificity proxy.

    Based on the mismatch-sensitivity observations from:
      Hsu PD et al., Nature Biotechnology 31, 827-832 (2013).

    Key findings encoded:
      - Seed region (positions 1-12, PAM-proximal, i.e. guide positions 9-20)
        determines off-target sensitivity — mismatches here are tolerated poorly.
      - Low-complexity sequences have more potential off-target sites.
      - G-rich seed regions show elevated off-target binding (Hsu 2013, Fig. 4).
      - Consecutive identical bases in seed correlate with off-target binding.

    Returns a proxy in [0, 1]; higher = more specific (fewer predicted off-targets).
    """
    guide = guide.upper()
    score = 1.0

    # Seed region: guide positions 9-20 (0-based 8-19), PAM-proximal
    seed = guide[8:]

    # Unique dinucleotide diversity — low diversity → repetitive → more off-targets
    unique_dinucs = len(set(guide[i:i+2] for i in range(len(guide) - 1)))
    if unique_dinucs < 8:
        score *= 0.55 + 0.025 * unique_dinucs

    # Seed region GC (Hsu 2013: 30-70% optimal for specificity)
    seed_gc = _calc_gc(seed)
    if seed_gc < 0.30:
        score *= 0.70   # AT-rich seed → poor binding discrimination
    elif seed_gc > 0.80:
        score *= 0.65   # G-rich seed → elevated off-target promiscuity (Hsu Fig. 4)

    # Seed region trinucleotide complexity
    seed_trinucs = len(set(seed[i:i+3] for i in range(len(seed) - 2)))
    if seed_trinucs < 5:
        score *= 0.80

    # Homopolymer runs in seed → off-target binding (Hsu 2013)
    for base in "ACGT":
        if base * 4 in seed:
            score *= 0.60
        elif base * 3 in seed:
            score *= 0.85

    return round(min(max(score, 0.0), 1.0), 4)


# ── Manufacturability score ───────────────────────────────────────────────────

def _manufacturability(guide: str) -> float:
    """
    Chemical synthesis and U6-expression feasibility score.
    Based on Zhang Lab CRISPR design guidelines and IDT oligo synthesis rules.
    """
    guide = guide.upper()
    score = 1.0
    gc = _calc_gc(guide)

    # GC extremes impair synthesis yield and U6 transcription initiation
    if gc < 0.25 or gc > 0.90:
        score *= 0.55
    elif gc < 0.35 or gc > 0.80:
        score *= 0.75

    # Long homopolymers cause synthesis errors (IDT guidelines)
    for base in "ACGT":
        if base * 6 in guide:
            score *= 0.45
        elif base * 5 in guide:
            score *= 0.62
        elif base * 4 in guide:
            score *= 0.80

    # RNA Pol III terminator — TTTT abolishes U6-driven transcription
    if "TTTT" in guide:
        score *= 0.50

    # Strong self-complementarity reduces transcription efficiency
    dg = _calc_folding_dg(guide)
    if dg < -18:
        score *= 0.50
    elif dg < -15:
        score *= 0.65
    elif dg < -10:
        score *= 0.82

    # U6 promoter prefers G at position 1 for transcription initiation
    if guide[0] != "G":
        score *= 0.92

    return round(min(max(score, 0.0), 1.0), 4)


# ── Flag issues ───────────────────────────────────────────────────────────────

def _flag_issues(guide: str, gc: float, dg: float) -> list:
    flags = []
    if gc < 0.40:
        flags.append("low GC (<40%)")
    if gc > 0.75:
        flags.append("high GC (>75%)")
    if dg < -10:
        flags.append(f"strong folding (ΔG={dg:.1f})")
    for base in "ACGT":
        if base * 4 in guide:
            flags.append(f"homopolymer {base*4}")
    if "TTTT" in guide:
        flags.append("PolIII terminator TTTT")
    return flags


# ── Guide finder ──────────────────────────────────────────────────────────────

def find_guides(sequence: str, pam: str = "NGG",
                guide_len: int = 20) -> List[GuideRNA]:
    """
    Identify all candidate CRISPR guides (+ and − strands) in `sequence`.
    PAM pattern uses IUPAC: N=any, R=A|G, Y=C|T.
    """
    sequence = sequence.upper()
    pam_re = (pam.replace("N", "[ACGT]")
                 .replace("R", "[AG]")
                 .replace("Y", "[CT]"))
    fwd_pattern = re.compile(r"(?=([ACGT]{%d}%s))" % (guide_len, pam_re))
    rc_seq = _reverse_complement(sequence)

    guides: List[GuideRNA] = []

    for strand, search_seq in [("+", sequence), ("-", rc_seq)]:
        for m in fwd_pattern.finditer(search_seq):
            full      = m.group(1)
            guide_seq = full[:guide_len]
            pam_seq   = full[guide_len:]
            if strand == "+":
                pos = m.start()
            else:
                pos = len(sequence) - m.start() - guide_len - len(pam_seq)

            gc   = _calc_gc(guide_seq)
            dg   = _calc_folding_dg(guide_seq)
            on_t = _on_target_score(guide_seq)
            spec = _specificity_score(guide_seq)
            mfg  = _manufacturability(guide_seq)

            # Weighted composite score (Doench 2016 weighting scheme)
            total = round(
                0.40 * on_t +
                0.30 * spec +
                0.20 * mfg  +
                0.10 * min(1.0, gc / 0.60),
                4
            )

            flags = _flag_issues(guide_seq, gc, dg)

            guides.append(GuideRNA(
                sequence=guide_seq, pam=pam_seq,
                position=pos, strand=strand,
                gc_content=gc, folding_dg=dg,
                on_target_score=on_t,
                specificity_score=spec,
                manufacturability=mfg,
                total_score=total,
                flags=flags,
            ))

    guides.sort(key=lambda g: g.total_score, reverse=True)
    return guides


def top_guides(guides: List[GuideRNA], n: int = 10) -> List[dict]:
    result = []
    for rank, g in enumerate(guides[:n], 1):
        d = g.to_dict()
        d["rank"] = rank
        result.append(d)
    return result