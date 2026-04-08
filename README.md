# 🧬 CRISPR Guide RNA Optimizer — OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v2.0-blue)](https://github.com/openenv-org)
[![Difficulty](https://img.shields.io/badge/Tasks-Easy%20%7C%20Medium%20%7C%20Hard-green)]()
[![Data](https://img.shields.io/badge/Data-NCBI%20RefSeq%20(live)-orange)]()

> **Real CRISPR science. Real NCBI sequences. Full OpenEnv agent API.**

An AI agent learns to design CRISPR-Cas9 guide RNAs for human disease genes.
Every sequence is fetched live from **NCBI E-utilities** (with bundled offline fallback).
Reward is shaped at every step using four published scoring models.

---

## What the Agent Must Do

The agent receives a real human mRNA sequence (TP53, BRCA1, ACE2, etc.) and must
propose 20-nucleotide guide RNA sequences, one per step. At each step the environment:

1. Scores the guide using **Doench 2016** (on-target activity), **Hsu 2013** (specificity),
   **SantaLucia 1998** thermodynamics (folding ΔG), and manufacturability heuristics.
2. Returns a **shaped reward** in `[-1, 1]` with partial credit at every step.
3. Updates the observation with the new guide's scores and running history.

---

## Three Tasks — Easy → Medium → Hard

| Task | Difficulty | Max Steps | Objective | Threshold |
|------|-----------|-----------|-----------|-----------|
| `single_guide_easy` | 🟢 Easy | 5 | Find one guide for TP53 with score ≥ 0.65 | 0.65 |
| `ranked_panel_medium` | 🟡 Medium | 10 | Build a diverse panel of 5 guides for BRCA1 (mean ≥ 0.60) | 0.60 |
| `multi_gene_hard` | 🔴 Hard | 15 | Design guides across TP53 → BRCA1 → ACE2 (fleet avg ≥ 0.70, penalised for quality flags) | 0.70 |

---

## Observation Space

```json
{
  "gene_name":          "TP53 (Tumour Suppressor p53)",
  "accession":          "NM_000546",
  "sequence":           "ATGGAGGAGCCGCAGTCAGA...",  // real NCBI mRNA region
  "region_start":       200,
  "region_length":      300,
  "pam":                "NGG",
  "step_number":        1,
  "max_steps":          5,
  "task_name":          "single_guide_easy",
  "task_description":   "Design one high-quality CRISPR guide...",
  "previous_guides":    [...],          // scored history of this episode
  "best_score_so_far":  0.73,
  "done":               false
}
```

## Action Space

```json
{
  "guide_sequence": "ATGGAGGAGCCGCAGTCAGA",   // exactly 20 nt, A/C/G/T only
  "rationale":      "Good GC content, no homopolymers"  // optional
}
```

## Reward Space

```json
{
  "step_reward":      0.78,     // shaped reward for this step [0, 1]
  "total_score":      0.73,     // composite Doench-weighted score [0, 1]
  "on_target":        0.81,     // Doench 2016 heuristic
  "specificity":      0.65,     // Hsu 2013 proxy
  "manufacturability": 0.88,   // synthesis feasibility
  "gc_pct":           55.0,     // GC content percentage
  "folding_dg":       -4.8,    // ΔG kcal/mol (SantaLucia 1998)
  "flags":            [],       // quality warnings
  "penalty":          0.0,      // deductions for invalid/duplicate input
  "episode_score":    null      // non-null only when done=True
}
```

---

## Quick Start

### Local (Python)

```bash
pip install -r requirements.txt
python server.py
# → http://localhost:7860
```

### Docker

```bash
docker build -t crispr-openenv .
docker run -p 7860:7860 crispr-openenv
```

### API

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "single_guide_easy"}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"guide_sequence": "ATGGAGGAGCCGCAGTCAGA", "rationale": "good GC"}'

# State
curl http://localhost:7860/state
```

### Run Inference Script

```bash
export HF_TOKEN=hf_yourtoken
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Reward Shaping Design

Reward is **dense** — the agent receives signal at every step:

| Condition | Reward Delta |
|-----------|-------------|
| Base: total_score | 0.0 – 1.0 |
| Guide clears task threshold | +0.10 |
| Guide score ≥ 0.90 | +0.20 |
| Guide improves best-so-far | +0.05 × Δ |
| PolIII terminator (TTTT) flag | −0.08 |
| GC out-of-range flag | −0.04 |
| Homopolymer / folding flag | −0.04 |
| Duplicate guide proposed | −0.15 |

---

## Baseline Scores

| Task | Baseline Model | Score | Steps |
|------|---------------|-------|-------|
| single_guide_easy | Qwen2.5-72B | ~0.72 | 3–4 |
| ranked_panel_medium | Qwen2.5-72B | ~0.55 | 8–10 |
| multi_gene_hard | Qwen2.5-72B | ~0.48 | 15 |

---

## Project Structure

```
crispr_openenv/
├── server.py                  ← FastAPI OpenEnv HTTP server (reset/step/state)
├── inference.py               ← Baseline inference script (mandatory)
├── openenv.yaml               ← OpenEnv spec metadata
├── Dockerfile                 ← HuggingFace Spaces deployment
├── requirements.txt
├── README.md
├── src/
│   ├── env.py                 ← CRISPREnv class (step/reset/state)
│   ├── models.py              ← Pydantic: Observation, Action, Reward
│   ├── tasks.py               ← 3 task configs + deterministic graders
│   ├── crispr_engine.py       ← Doench 2016 scoring engine (original)
│   ├── ncbi_fetcher.py        ← Live NCBI API + offline fallback (original)
│   └── bundled_sequences.py   ← Real NCBI sequences for offline use (original)
├── tests/
│   └── test_env.py            ← Unit tests for graders, models, API
└── outputs/                   ← Sample JSON outputs
```

---

## Available Genes

| Gene | Accession | Disease |
|------|-----------|---------|
| TP53 | NM_000546 | Pan-cancer tumour suppressor |
| BRCA1 | NM_007294 | Breast/ovarian cancer |
| CFTR | NM_000492 | Cystic fibrosis |
| HTT | NM_002111 | Huntington disease |
| EGFR | NM_005228 | Lung cancer |
| VEGFA | NM_001025366 | Angiogenesis |
| TNF | NM_000594 | Inflammation |
| IL6 | NM_000600 | Cytokine storm |
| APOE | NM_000041 | Alzheimer's disease |
| ACE2 | NM_021804 | SARS-CoV-2 receptor |

---

## Scoring Model References

- **On-target (40%)** — Doench et al. 2016, *Nature Biotechnology*
- **Specificity (30%)** — Hsu et al. 2013, *Nature Biotechnology*
- **Manufacturability (20%)** — Zhang Lab CRISPR guidelines
- **GC bonus (10%)** — Established best practice (40–75% optimal)
- **Thermodynamics** — SantaLucia 1998, nearest-neighbour ΔG
