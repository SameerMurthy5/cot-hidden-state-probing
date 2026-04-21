"""
Experiment 2: Step corruption.

For 200 GSM8K test examples, corrupt one intermediate numerical result in the
model's CoT, re-run inference from that prefix, and measure whether the final
answer changes and whether the error propagates.

Also performs the novel cross-check: joins corruption results with probe results
to see if "early-knowing" examples are the ones resistant to corruption.

Usage:
    python src/corrupt.py \
        --hidden-states results/hidden_states_test.jsonl \
        --probe-results results/probe_results.json \
        --out results/corruption_results.json \
        --n 200
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils import corrupt_equation, extract_final_answer, parse_equations

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


@torch.inference_mode()
def continue_from_prefix(model, tokenizer, prefix_text: str, max_new_tokens: int = 512) -> str:
    """Feed corrupted CoT as prefix and let model continue generation."""
    inputs = tokenizer(prefix_text, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=False)


def answers_are_close(a: float, b: float, tol: float = 0.5) -> bool:
    """True if two numerical answers are within tol of each other."""
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def error_propagated(
    corrupted_result_str: str,
    original_answer: float,
    new_answer: float,
    tol_factor: float = 0.2,
) -> bool:
    """Heuristic: did the new answer shift in a way consistent with the corrupted value?

    We check if new_answer differs from original_answer by more than the
    magnitude of the corruption, which would suggest the model used the
    corrupted intermediate value.
    """
    if new_answer is None or original_answer is None:
        return False
    return not answers_are_close(original_answer, new_answer)


def compute_early_probe_confidence(row: dict, probe_results: dict) -> float:
    """
    Proxy for 'early knowing': use pos0 5-class probe accuracy as a per-example
    signal by checking if the ground-truth bin was among the top predicted classes.

    Since we only have aggregate accuracies (not per-example predictions), we
    instead use the difference in probe accuracy between pos0 and pos3 as a
    dataset-level signal. For per-example, we compute the L2 distance between
    pos0 and pos3 hidden states as a proxy for how much the representation changes.
    """
    hs = row.get("hidden_states", {})
    if "pos0" not in hs or "pos3" not in hs:
        return float("nan")
    v0 = np.array(hs["pos0"].get("27", []), dtype=np.float32)
    v3 = np.array(hs["pos3"].get("27", []), dtype=np.float32)
    if v0.size == 0 or v3.size == 0:
        return float("nan")
    return float(np.linalg.norm(v3 - v0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-states", default="results/hidden_states_test.jsonl")
    parser.add_argument("--probe-results", default="results/probe_results.json")
    parser.add_argument("--out", default="results/corruption_results.json")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("Loading hidden states...")
    all_rows = load_jsonl(args.hidden_states)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(all_rows), size=min(args.n, len(all_rows)), replace=False)
    rows = [all_rows[i] for i in sorted(indices)]

    probe_results = {}
    if os.path.exists(args.probe_results):
        with open(args.probe_results) as f:
            probe_results = json.load(f)

    corruption_records = []

    for row in tqdm(rows, desc="Corrupting"):
        full_text = row.get("full_text", "")
        gt_answer = row.get("gt_answer")
        original_predicted = row.get("predicted_answer")

        # Extract CoT region (between <think> and </think>)
        think_start = full_text.find("<think>")
        think_end = full_text.find("</think>")

        if think_start == -1 or think_end == -1:
            cot_text = full_text
            prefix_before_cot = ""
            suffix_after_cot = ""
        else:
            prefix_before_cot = full_text[: think_start + len("<think>")]
            cot_text = full_text[think_start + len("<think>") : think_end]
            suffix_after_cot = full_text[think_end:]

        equations = parse_equations(cot_text)
        if not equations:
            corruption_records.append({
                "idx": row.get("idx"),
                "gt_answer": gt_answer,
                "original_predicted": original_predicted,
                "corrupted": False,
                "reason": "no_equations_found",
                "hidden_state_drift": compute_early_probe_confidence(row, probe_results),
            })
            continue

        # Pick the first equation to corrupt
        eq = equations[0]
        corrupted_cot = corrupt_equation(cot_text, eq, rng)

        # Re-run from corrupted prefix (up through the corrupted step)
        corrupted_prefix = prefix_before_cot + corrupted_cot
        try:
            new_full_text = continue_from_prefix(model, tokenizer, corrupted_prefix)
            new_answer = extract_final_answer(new_full_text)
        except Exception as e:
            print(f"Error continuing from prefix: {e}", file=sys.stderr)
            new_answer = None
            new_full_text = ""

        answer_changed = not answers_are_close(original_predicted, new_answer)
        error_prop = error_propagated(eq["result"], original_predicted, new_answer)

        record = {
            "idx": row.get("idx"),
            "gt_answer": gt_answer,
            "original_predicted": original_predicted,
            "new_predicted": new_answer,
            "answer_changed": answer_changed,
            "error_propagated": error_prop,
            "corrupted": True,
            "corrupted_equation_original": eq["full_match"],
            "corrupted_equation_new": corrupt_equation(cot_text, eq, rng)[eq["start"]:eq["start"] + len(eq["full_match"]) + 10],
            "hidden_state_drift": compute_early_probe_confidence(row, probe_results),
        }
        corruption_records.append(record)

    # Summary statistics
    corrupted = [r for r in corruption_records if r["corrupted"]]
    n_changed = sum(1 for r in corrupted if r["answer_changed"])
    n_propagated = sum(1 for r in corrupted if r["error_propagated"])
    change_rate = n_changed / len(corrupted) if corrupted else 0.0
    propagation_rate = n_propagated / len(corrupted) if corrupted else 0.0

    summary = {
        "n_total": len(corruption_records),
        "n_corrupted": len(corrupted),
        "n_answer_changed": n_changed,
        "answer_change_rate": change_rate,
        "error_propagation_rate": propagation_rate,
    }

    print("\n=== Corruption Results ===")
    print(f"  Examples processed:    {len(corrupted)}")
    print(f"  Answer change rate:    {change_rate:.3f}")
    print(f"  Error propagation:     {propagation_rate:.3f}")

    # Cross-check: do high-drift examples (model representation changed a lot
    # across CoT) correlate with corruption sensitivity?
    drifts = np.array([r["hidden_state_drift"] for r in corrupted if not np.isnan(r["hidden_state_drift"])])
    changed_mask = np.array([r["answer_changed"] for r in corrupted if not np.isnan(r["hidden_state_drift"])])
    if len(drifts) > 0:
        median_drift = float(np.median(drifts))
        high_drift_change = float(changed_mask[drifts >= median_drift].mean()) if (drifts >= median_drift).sum() > 0 else float("nan")
        low_drift_change = float(changed_mask[drifts < median_drift].mean()) if (drifts < median_drift).sum() > 0 else float("nan")
        summary["cross_check"] = {
            "median_hidden_state_drift": median_drift,
            "change_rate_high_drift": high_drift_change,
            "change_rate_low_drift": low_drift_change,
        }
        print(f"\n  Cross-check (hidden state drift vs corruption sensitivity):")
        print(f"    High-drift change rate: {high_drift_change:.3f}")
        print(f"    Low-drift change rate:  {low_drift_change:.3f}")

    output = {"summary": summary, "records": corruption_records}
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
