"""
Train and evaluate logistic regression probes on hidden states from generate_cot.py output.

Usage:
    python src/probe.py --train results/hidden_states_train.jsonl \
                        --test  results/hidden_states_test.jsonl  \
                        --out   results/probe_results.json
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import bin_answers, bin_answers_binary, extract_final_answer

POSITIONS = ["pos0", "pos1", "pos2", "pos3"]
POSITION_LABELS = {
    "pos0": "After prompt",
    "pos1": "After step 1",
    "pos2": "CoT midpoint",
    "pos3": "Before answer",
}
LAYERS = [27, 12]
N_BINS = 5


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_feature_matrix(rows: list[dict], pos: str, layer: int) -> np.ndarray:
    X = []
    for row in rows:
        vec = row["hidden_states"][pos][str(layer)]
        X.append(vec)
    return np.array(X, dtype=np.float32)


def get_gt_labels(rows: list[dict], n_bins: int = N_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Return (5-class labels, binary labels) for ground-truth answers."""
    answers = [row["gt_answer"] for row in rows]
    labels_5 = bin_answers(answers, n_bins=n_bins)
    labels_2 = bin_answers_binary(answers)
    return labels_5, labels_2


def train_probe(X_train, y_train, X_test, y_test, random_state=42) -> dict:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_s))

    # Random-label baseline
    y_shuffled = y_train.copy()
    rng = np.random.default_rng(random_state)
    rng.shuffle(y_shuffled)
    clf_rand = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state, n_jobs=-1)
    clf_rand.fit(X_train_s, y_shuffled)
    rand_acc = accuracy_score(y_test, clf_rand.predict(X_test_s))

    # Majority-class baseline
    majority_class = np.bincount(y_train).argmax()
    maj_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))

    return {
        "accuracy": float(acc),
        "random_baseline": float(rand_acc),
        "majority_baseline": float(maj_acc),
        "clf": clf,
        "scaler": scaler,
    }


def run_probing(train_rows: list[dict], test_rows: list[dict]) -> dict:
    labels_5_train, labels_2_train = get_gt_labels(train_rows)
    labels_5_test, labels_2_test = get_gt_labels(test_rows)

    results = {}

    for layer in LAYERS:
        layer_key = f"layer{layer}"
        results[layer_key] = {}

        for pos in POSITIONS:
            X_train = build_feature_matrix(train_rows, pos, layer)
            X_test = build_feature_matrix(test_rows, pos, layer)

            # 5-class probe
            r5 = train_probe(X_train, labels_5_train, X_test, labels_5_test)
            # Binary probe
            r2 = train_probe(X_train, labels_2_train, X_test, labels_2_test)

            results[layer_key][pos] = {
                "5class": {k: v for k, v in r5.items() if k not in ("clf", "scaler")},
                "binary": {k: v for k, v in r2.items() if k not in ("clf", "scaler")},
                "_clf_5class": r5["clf"],
                "_scaler_5class": r5["scaler"],
                "_clf_binary": r2["clf"],
                "_scaler_binary": r2["scaler"],
            }

            print(
                f"  layer={layer:2d}  {pos}  "
                f"5-class acc={r5['accuracy']:.3f}  "
                f"(rand={r5['random_baseline']:.3f}, maj={r5['majority_baseline']:.3f})  "
                f"| binary acc={r2['accuracy']:.3f}"
            )

    return results


def save_results(results: dict, out_path: str, probes_dir: str):
    os.makedirs(probes_dir, exist_ok=True)
    serializable = {}
    for layer_key, layer_data in results.items():
        serializable[layer_key] = {}
        for pos, pos_data in layer_data.items():
            serializable[layer_key][pos] = {
                "5class": pos_data["5class"],
                "binary": pos_data["binary"],
            }
            # Save sklearn objects
            for variant in ("5class", "binary"):
                probe_path = os.path.join(probes_dir, f"clf_{layer_key}_{pos}_{variant}.pkl")
                with open(probe_path, "wb") as f:
                    pickle.dump(
                        {
                            "clf": pos_data[f"_clf_{variant}"],
                            "scaler": pos_data[f"_scaler_{variant}"],
                        },
                        f,
                    )

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Probe results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="results/hidden_states_train.jsonl")
    parser.add_argument("--test", default="results/hidden_states_test.jsonl")
    parser.add_argument("--out", default="results/probe_results.json")
    parser.add_argument("--probes-dir", default="results/probes")
    args = parser.parse_args()

    print("Loading data...")
    train_rows = load_jsonl(args.train)
    test_rows = load_jsonl(args.test)
    print(f"  Train: {len(train_rows)} examples, Test: {len(test_rows)} examples")

    # Filter rows missing hidden states for any position
    def has_all_positions(row):
        return all(
            pos in row.get("hidden_states", {}) for pos in POSITIONS
        ) and row.get("gt_answer") is not None

    train_rows = [r for r in train_rows if has_all_positions(r)]
    test_rows = [r for r in test_rows if has_all_positions(r)]
    print(f"  After filtering: Train={len(train_rows)}, Test={len(test_rows)}")

    print("\nTraining probes...")
    results = run_probing(train_rows, test_rows)
    save_results(results, args.out, args.probes_dir)


if __name__ == "__main__":
    main()
