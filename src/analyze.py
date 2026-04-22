"""
Generate all result figures and print summary tables.

Usage:
    python src/analyze.py \
        --probe-results results/probe_results.json \
        --corruption-results results/corruption_results.json \
        --hidden-states results/hidden_states_test.jsonl \
        --out-dir results/figures
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams["figure.dpi"] = 120

POSITIONS = ["pos0", "pos1", "pos2", "pos3"]
POSITION_LABELS = ["After prompt", "After step 1", "CoT midpoint", "Before answer"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def plot_probe_accuracy(probe_data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, variant, title in zip(
        axes,
        ["5class", "binary"],
        ["5-class probe (equal-freq bins)", "Binary probe (above/below median)"],
    ):
        for layer_key, label in [("layer27", "Last layer (27)"), ("layer12", "Middle layer (12)")]:
            accs = [probe_data[layer_key][pos][variant]["accuracy"] for pos in POSITIONS]
            ax.plot(POSITION_LABELS, accs, marker="o", label=label)

        rand_base = probe_data["layer27"]["pos0"][variant]["random_baseline"]
        maj_base = probe_data["layer27"]["pos0"][variant]["majority_baseline"]
        ax.axhline(rand_base, linestyle="--", color="gray", alpha=0.7, label=f"Random ({rand_base:.2f})")
        ax.axhline(maj_base, linestyle=":", color="black", alpha=0.7, label=f"Majority ({maj_base:.2f})")

        ax.set_title(title)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Probe Accuracy at Each CoT Position", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "probe_accuracy_curve.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def print_probe_table(probe_data):
    rows = []
    for layer_key in ["layer27", "layer12"]:
        for pos, pos_label in zip(POSITIONS, POSITION_LABELS):
            d = probe_data[layer_key][pos]
            rows.append({
                "Layer": layer_key,
                "Position": pos_label,
                "5-class acc": d["5class"]["accuracy"],
                "Binary acc": d["binary"]["accuracy"],
                "Random base": d["5class"]["random_baseline"],
                "Majority base": d["5class"]["majority_baseline"],
            })
    df = pd.DataFrame(rows)
    print("\n=== Probe Accuracy Table ===")
    print(df.to_string(index=False, float_format="{:.3f}".format))


def plot_corruption(summary, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Answer\nchanged", "Error\npropagated", "Answer\nunchanged"]
    vals = [
        summary["answer_change_rate"],
        summary["error_propagation_rate"],
        1 - summary["answer_change_rate"],
    ]
    bars = ax.bar(labels, vals, color=["#e07b54", "#e04444", "#54ae7b"])
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate")
    ax.set_title("Step Corruption Experiment Results", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "corruption_bar.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_crosscheck(summary, records, out_dir):
    cc = summary.get("cross_check")
    if not cc:
        print("No cross-check data available.")
        return

    df = pd.DataFrame([
        {
            "hidden_state_drift": r["hidden_state_drift"],
            "answer_changed": int(r["answer_changed"]),
        }
        for r in records
        if r["corrupted"] and r.get("hidden_state_drift") is not None
        and not (isinstance(r["hidden_state_drift"], float) and np.isnan(r["hidden_state_drift"]))
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    labels = ["Low drift\n(pre-committed)", "High drift\n(reasoning-dependent)"]
    vals = [cc["change_rate_low_drift"], cc["change_rate_high_drift"]]
    ax.bar(labels, vals, color=["#54ae7b", "#e07b54"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Answer change rate under corruption")
    ax.set_title("Change Rate by Hidden State Drift Group", fontweight="bold")

    ax2 = axes[1]
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.02, 0.02, size=len(df))
    ax2.scatter(df["hidden_state_drift"], df["answer_changed"] + jitter, alpha=0.4, s=15)
    ax2.set_xlabel("Hidden state drift (L2 norm, pos0→pos3, layer 27)")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Unchanged", "Changed"])
    ax2.set_title("Per-example: Drift vs Corruption Effect", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(out_dir, "crosscheck.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def print_task_accuracy(test_rows):
    correct = sum(
        1 for r in test_rows
        if r.get("predicted_answer") is not None
        and r.get("gt_answer") is not None
        and abs(r["predicted_answer"] - r["gt_answer"]) <= 0.5
    )
    total = len(test_rows)
    print(f"\nGSM8K task accuracy: {correct}/{total} = {correct / total:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-results", default="results/probe_results.json")
    parser.add_argument("--corruption-results", default="results/corruption_results.json")
    parser.add_argument("--hidden-states", default="results/hidden_states_test.jsonl")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.probe_results) as f:
        probe_data = json.load(f)

    with open(args.corruption_results) as f:
        corruption_data = json.load(f)

    test_rows = load_jsonl(args.hidden_states)

    print_task_accuracy(test_rows)
    print_probe_table(probe_data)

    summary = corruption_data["summary"]
    print(f"\n=== Corruption Summary ===")
    print(f"  Examples:           {summary['n_corrupted']}")
    print(f"  Answer change rate: {summary['answer_change_rate']:.3f}")
    print(f"  Error propagation:  {summary['error_propagation_rate']:.3f}")

    plot_probe_accuracy(probe_data, args.out_dir)
    plot_corruption(summary, args.out_dir)
    plot_crosscheck(summary, corruption_data["records"], args.out_dir)


if __name__ == "__main__":
    main()
