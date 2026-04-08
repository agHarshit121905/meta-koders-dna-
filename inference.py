"""
inference.py — OpenEnv Hackathon Baseline Inference Script
===========================================================

Runs an LLM agent (via OpenAI client) against all 3 CRISPR tasks and
emits the mandatory [START] / [STEP] / [END] stdout format.

Environment variables
---------------------
API_BASE_URL   LLM endpoint  (default: https://router.huggingface.co/v1)
MODEL_NAME     Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
HF_TOKEN       API key
"""

import os
import sys
import json
import re
import time
from typing import List, Optional

from openai import OpenAI

# ── Add src to path so we can import the env ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from env import CRISPREnv
from models import CRISPRAction
from tasks import TASKS

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") 
BENCHMARK    = "crispr-guide-optimizer"

# Force bundled sequences for reproducible offline runs (set OFFLINE_MODE=1)
os.environ.setdefault("OFFLINE_MODE", "1")

TASKS_TO_RUN = [
    "single_guide_easy",
    "ranked_panel_medium",
    "multi_gene_hard",
]

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ── Prompt builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert molecular biologist specialising in CRISPR-Cas9 guide RNA design.
You will be given a DNA sequence and must propose exactly one 20-nucleotide guide RNA per response.

Rules:
1. Output ONLY a JSON object — nothing else. No prose, no markdown.
2. JSON format: {"guide_sequence": "<20 nt>", "rationale": "<brief reason>"}
3. guide_sequence must be exactly 20 characters using only A, C, G, T (uppercase).
4. Pick a subsequence that:
   - Has GC content between 40–75%
   - Avoids homopolymer runs (AAAA, CCCC, GGGG, TTTT)
   - Avoids the TTTT motif (PolIII terminator)
   - Follows the NGG PAM convention (the guide should NOT include the PAM)
5. For the medium task, make each guide as different as possible from previous ones.
6. For the hard task, adapt your strategy to each gene as it changes.
"""

def build_user_prompt(obs_dict: dict) -> str:
    seq = obs_dict.get("sequence", "")
    prev = obs_dict.get("previous_guides", [])
    prev_seqs = [g.get("guide_sequence", g.get("sequence", "")) for g in prev[-5:]]

    prompt = (
        f"Task: {obs_dict.get('task_name')}\n"
        f"Gene: {obs_dict.get('gene_name')} ({obs_dict.get('accession')})\n"
        f"PAM: {obs_dict.get('pam')}\n"
        f"Step: {obs_dict.get('step_number')} / {obs_dict.get('max_steps')}\n"
        f"Best score so far: {obs_dict.get('best_score_so_far', 0):.3f}\n\n"
        f"DNA sequence ({obs_dict.get('region_length')} bp):\n{seq}\n\n"
        f"Objective: {obs_dict.get('task_description', '')[:300]}\n"
    )
    if prev_seqs:
        prompt += f"\nAlready proposed (avoid duplicates):\n" + "\n".join(prev_seqs) + "\n"
    prompt += "\nPropose your next guide RNA as JSON:"
    return prompt


# ── LLM call with fallback ────────────────────────────────────────────────────

def _extract_guide_from_sequence(sequence: str) -> str:
    """Fallback: pick the first 20-nt window with good GC from the sequence."""
    seq = sequence.upper()
    best_seq = seq[:20]
    best_gc  = 0.0
    for i in range(0, len(seq) - 20, 5):
        window = seq[i:i+20]
        if len(window) < 20:
            break
        gc = (window.count("G") + window.count("C")) / 20
        if 0.40 <= gc <= 0.75 and "TTTT" not in window:
            if gc > best_gc:
                best_gc  = gc
                best_seq = window
    return best_seq


def call_llm(obs_dict: dict, last_error: Optional[str] = None) -> CRISPRAction:
    """Call the LLM and parse its guide RNA proposal. Falls back on parse failure."""
    user_msg = build_user_prompt(obs_dict)
    if last_error:
        user_msg += f"\n\n[CORRECTION NEEDED] Previous guide was invalid: {last_error}. Try a different sequence."

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            parsed = json.loads(m.group()) if m else {}

        guide_seq = parsed.get("guide_sequence", "").upper().strip()
        rationale = parsed.get("rationale", "")

        if len(guide_seq) != 20 or not re.match(r'^[ACGT]{20}$', guide_seq):
            raise ValueError(f"Bad guide from LLM: '{guide_seq}'")

        return CRISPRAction(guide_sequence=guide_seq, rationale=rationale)

    except Exception:
        seq = obs_dict.get("sequence", "A" * 20)
        fb_seq = _extract_guide_from_sequence(seq)
        return CRISPRAction(guide_sequence=fb_seq, rationale="fallback-heuristic")


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_name: str, tasks: dict) -> dict:
    """Run one full episode and return summary."""
    task_config = tasks[task_name]
    env   = CRISPREnv(task_name=task_name)
    obs   = env.reset()

    rewards: List[float]      = []
    last_error: Optional[str]  = None
    final_score: float         = 0.0
    done: bool                 = False
    step_n: int                = 0

    obs_dict = obs.model_dump()

    # ── [START] ──────────────────────────────────────────────────────────
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        while not done and step_n < obs.max_steps:
            action = call_llm(obs_dict, last_error)
            last_error = None

            try:
                obs, reward, done, info = env.step(action)
                obs_dict   = obs.model_dump()
                step_n    += 1
                step_rew   = reward.step_reward
                rewards.append(step_rew)

                if reward.episode_score is not None:
                    final_score = reward.episode_score

                # error= is strictly for action exceptions per spec
                # warnings/notes go to info but are not emitted in error= field
                last_error = None

                # ── [STEP] ────────────────────────────────────────────────
                print(
                    f"[STEP] step={step_n} "
                    f"action={action.guide_sequence} "
                    f"reward={step_rew:.2f} "
                    f"done={'true' if done else 'false'} "
                    f"error=null",
                    flush=True,
                )

            except Exception as e:
                last_error = str(e)
                step_n    += 1
                rewards.append(-0.30)
                print(
                    f"[STEP] step={step_n} "
                    f"action={action.guide_sequence} "
                    f"reward=-0.30 "
                    f"done=false "
                    f"error={last_error}",
                    flush=True,
                )

            time.sleep(0.2)   # rate limit courtesy

    finally:
        env.close()

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        success     = final_score >= task_config.success_threshold
        steps_taken = step_n

        # ── [END] ─────────────────────────────────────────────────────────
        print(
            f"[END] "
            f"success={'true' if success else 'false'} "
            f"steps={steps_taken} "
            f"score={final_score:.2f} "
            f"rewards={rewards_str}",
            flush=True,
        )

    return {
        "task": task_name,
        "score": final_score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = []
    for task_name in TASKS_TO_RUN:
        print(f"\n{'='*60}", flush=True)
        result = run_episode(task_name, TASKS)
        results.append(result)
        print(f"{'='*60}", flush=True)
        time.sleep(1.0)   # pause between tasks

    # Summary
    print("\n\n[SUMMARY]", flush=True)
    for r in results:
        print(
            f"  task={r['task']} score={r['score']:.2f} "
            f"success={'true' if r['success'] else 'false'} steps={r['steps']}",
            flush=True,
        )
    avg = sum(r["score"] for r in results) / len(results)
    print(f"  average_score={avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
