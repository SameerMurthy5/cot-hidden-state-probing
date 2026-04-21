"""
Generate CoT traces and extract hidden states at 4 positions for GSM8K problems.

Usage:
    python src/generate_cot.py --split test --n 1319 --out results/hidden_states_test.jsonl
    python src/generate_cot.py --split train --n 500  --out results/hidden_states_train.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils import extract_final_answer, find_cot_positions

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
FALLBACK_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

LAST_LAYER = 27   # 0-indexed; 1.5B model has 28 layers (0–27)
MIDDLE_LAYER = 12


def build_prompt(problem: str, tokenizer) -> str:
    messages = [{"role": "user", "content": problem}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def extract_hidden_at_position(
    model,
    input_ids: torch.Tensor,
    token_idx: int,
    layers: list[int],
) -> dict[int, list[float]]:
    """Forward pass on input_ids[:token_idx+1], return hidden states from specified layers."""
    prefix = input_ids[:, : token_idx + 1]
    out = model(prefix, output_hidden_states=True, use_cache=False)
    result = {}
    for layer in layers:
        # hidden_states is tuple of (n_layers+1) tensors, shape (1, seq, hidden)
        vec = out.hidden_states[layer + 1][:, -1, :].squeeze(0)
        result[layer] = vec.float().cpu().numpy().tolist()
    return result


@torch.inference_mode()
def generate_cot(model, tokenizer, prompt: str, max_new_tokens: int = 2048) -> tuple[str, torch.Tensor]:
    """Generate CoT and return (full_text, full_token_ids tensor)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    return full_text, output_ids


def process_example(model, tokenizer, problem: str, gt_answer: float, layers: list[int]):
    prompt = build_prompt(problem, tokenizer)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]

    full_text, output_ids = generate_cot(model, tokenizer, prompt)
    output_ids = output_ids.to(model.device)

    predicted_answer = extract_final_answer(full_text)

    positions = find_cot_positions(output_ids[0].tolist(), prompt_len, tokenizer)

    hidden_states = {}
    for pos_name, tok_idx in positions.items():
        hidden_states[pos_name] = extract_hidden_at_position(model, output_ids, tok_idx, layers)

    return {
        "prompt": prompt,
        "full_text": full_text,
        "predicted_answer": predicted_answer,
        "gt_answer": gt_answer,
        "positions": positions,
        "hidden_states": {
            pos: {str(layer): vecs for layer, vecs in layer_dict.items()}
            for pos, layer_dict in hidden_states.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--n", type=int, default=None, help="Max examples to process")
    parser.add_argument("--out", default=None)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Skip already-processed examples")
    args = parser.parse_args()

    out_path = args.out or f"results/hidden_states_{args.split}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading GSM8K ({args.split} split)")
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)

    rng = np.random.default_rng(args.seed)
    if args.n and args.n < len(dataset):
        indices = rng.choice(len(dataset), size=args.n, replace=False).tolist()
        dataset = dataset.select(sorted(indices))

    # Resume support: skip already-written lines
    already_done = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                row = json.loads(line)
                already_done.add(row.get("idx"))

    layers = [LAST_LAYER, MIDDLE_LAYER]

    with open(out_path, "a" if args.resume else "w") as fout:
        for i, example in enumerate(tqdm(dataset, desc="Generating")):
            if i in already_done:
                continue

            gt_answer = extract_final_answer(example["answer"])
            if gt_answer is None:
                continue

            try:
                result = process_example(model, tokenizer, example["question"], gt_answer, layers)
                result["idx"] = i
                fout.write(json.dumps(result) + "\n")
                fout.flush()
            except Exception as e:
                print(f"Error on example {i}: {e}", file=sys.stderr)
                continue

    print(f"Done. Output: {out_path}")


if __name__ == "__main__":
    main()
