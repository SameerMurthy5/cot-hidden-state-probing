import re
import numpy as np
from typing import Optional


def extract_final_answer(text: str) -> Optional[float]:
    """Pull the final numerical answer from model-generated CoT text.

    Handles DeepSeek-R1 format where answer appears after </think>, and
    GSM8K ground-truth format (#### N).
    """
    text = clean_bpe_artifacts(text)

    # GSM8K ground truth format
    gt_match = re.search(r"####\s*([\d,\.\-]+)", text)
    if gt_match:
        return _parse_number(gt_match.group(1))

    # DeepSeek-R1 boxed answer format: \boxed{18}
    boxed_match = re.search(r"\\boxed\{([\d,\.\-]+)\}", text)
    if boxed_match:
        return _parse_number(boxed_match.group(1))

    # After </think> tag — take last number in remaining text
    think_split = text.split("</think>")
    search_text = think_split[-1] if len(think_split) > 1 else text

    # Find all numbers in the text, return the last one
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", search_text)
    if numbers:
        return _parse_number(numbers[-1])

    return None


def _parse_number(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def bin_answers(answers: list[float], n_bins: int = 5) -> np.ndarray:
    """Assign equal-frequency bin labels to a list of numerical answers."""
    arr = np.array(answers, dtype=float)
    quantiles = np.linspace(0, 100, n_bins + 1)
    boundaries = np.percentile(arr, quantiles)
    # Use digitize; clip so indices are in [0, n_bins-1]
    labels = np.digitize(arr, boundaries[1:-1])
    return labels


def bin_answers_binary(answers: list[float]) -> np.ndarray:
    """Binary label: 1 if above median, 0 otherwise."""
    arr = np.array(answers, dtype=float)
    median = np.median(arr)
    return (arr > median).astype(int)


def find_cot_positions(full_token_ids: list[int], prompt_len: int, tokenizer) -> dict[str, int]:
    """Return token indices for the 4 probe extraction positions.

    Positions:
      pos0 — end of prompt (last prompt token)
      pos1 — after first reasoning step (first newline after <think>)
      pos2 — midpoint of CoT
      pos3 — just before </think>

    Returns token indices (into full_token_ids) for the last token of each prefix.
    """
    full_text = tokenizer.decode(full_token_ids, skip_special_tokens=False)

    # Locate <think> and </think> character spans
    think_start_char = full_text.find("<think>")
    think_end_char = full_text.find("</think>")

    # Convert character positions to approximate token positions using tokenizer
    # We binary-search by encoding substrings
    def char_to_token_idx(char_pos: int) -> int:
        prefix = full_text[:char_pos]
        toks = tokenizer.encode(prefix, add_special_tokens=False)
        return max(0, len(toks) - 1)

    pos0 = prompt_len - 1

    if think_start_char == -1 or think_end_char == -1:
        # Fallback: divide generation into quarters
        gen_len = len(full_token_ids) - prompt_len
        pos1 = prompt_len + max(0, gen_len // 4)
        pos2 = prompt_len + max(0, gen_len // 2)
        pos3 = len(full_token_ids) - 2
    else:
        think_open_tok = char_to_token_idx(think_start_char + len("<think>"))
        think_close_tok = char_to_token_idx(think_end_char)

        # First newline after <think>
        first_nl = full_text.find("\n", think_start_char + len("<think>"))
        if first_nl != -1 and first_nl < think_end_char:
            pos1 = char_to_token_idx(first_nl)
        else:
            pos1 = think_open_tok + 1

        pos2 = (think_open_tok + think_close_tok) // 2
        pos3 = max(think_close_tok - 1, pos2 + 1)

    # Ensure monotonically increasing and within bounds
    n = len(full_token_ids)
    positions = {"pos0": pos0, "pos1": pos1, "pos2": pos2, "pos3": pos3}
    for k, v in positions.items():
        positions[k] = int(np.clip(v, 0, n - 1))

    return positions


def clean_bpe_artifacts(text: str) -> str:
    """Replace GPT-2 BPE special characters with normal ASCII equivalents.

    The tokenizer encodes space as Ġ (U+0120) and newline as Ċ (U+010A).
    Python regex \\s does not match these, so clean them before any text processing.
    """
    return text.replace("\u0120", " ").replace("\u010a", "\n")


def parse_equations(cot_text: str) -> list[dict]:
    """Find arithmetic equations in CoT text.

    Handles DeepSeek-R1 output formats:
      16 - 3 = **13 eggs**      (markdown bold, trailing text)
      9 × $2 = **$18**          (dollar signs, × operator)
      24 * 3 = 72               (plain)
      Eggs kept = 16 - 3 = **13 eggs**  (labelled equations)
    """
    cot_text = clean_bpe_artifacts(cot_text)
    # Match: number op number = optional(** $) number optional(text **)
    pattern = re.compile(
        r"(\$?[\d,]+(?:\.\d+)?)"          # left operand (optional $)
        r"\s*[×x\*\+\-\/÷]\s*"            # operator
        r"(\$?[\d,]+(?:\.\d+)?)"          # right operand
        r"\s*=\s*"                         # equals
        r"\*{0,2}\$?"                      # optional markdown bold + dollar
        r"([\d,]+(?:\.\d+)?)",            # result number
        re.UNICODE,
    )
    results = []
    for m in pattern.finditer(cot_text):
        results.append({
            "full_match": m.group(0),
            "result": m.group(3),
            "start": m.start(),
            "end": m.end(),
        })
    return results


def corrupt_equation(cot_text: str, equation: dict, rng: np.random.Generator) -> str:
    """Replace the result of an equation with a wrong value within ±50%."""
    original = _parse_number(equation["result"])
    if original is None or original == 0:
        return cot_text

    delta = rng.uniform(0.1, 0.5) * abs(original)
    sign = rng.choice([-1, 1])
    corrupted = original + sign * delta

    # Round to integer if original was integer
    if "." not in equation["result"]:
        corrupted = int(round(corrupted))
        if corrupted == int(original):
            corrupted += sign * max(1, int(abs(original) * 0.1))
        corrupted_str = str(corrupted)
    else:
        corrupted_str = f"{corrupted:.2f}"

    new_match = equation["full_match"].replace(equation["result"], corrupted_str, 1)
    return cot_text[:equation["start"]] + new_match + cot_text[equation["end"]:]
