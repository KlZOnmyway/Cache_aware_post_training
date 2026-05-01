"""One-shot conversion utilities: dump MATH / MMLU prompts to jsonl for slime.

slime's ``--prompt-data`` flag expects a jsonl file with one record per line.
We re-use the parsing logic from the legacy rl_moe codebase
(``train_controller_standalone.py:collect_math_prompts`` etc.) but emit jsonl
rather than feed the dataset in-process.

These functions are run **once**, offline, before training. They are not
called inside the trainer.

Usage::

    python -m slime_adapter.data.prompt_loader \
        --task math --in /scratch/.../hendrycks_math --out /scratch/.../math.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List


def dump_math_jsonl(in_dir: Path, out_path: Path, max_prompt_len: int = 512) -> int:
    """Dump Hendrycks MATH prompts to jsonl. Returns number of records written.

    Each record: ``{"prompt": str, "label": str}``  — ``label`` is the boxed answer.
    """
    import pandas as pd  # local import; only needed for this offline step

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for cat_dir in sorted(p for p in in_dir.iterdir() if p.is_dir()):
            for parquet_path in sorted(cat_dir.glob("train-*.parquet")):
                df = pd.read_parquet(parquet_path)
                if not {"problem", "solution"}.issubset(df.columns):
                    continue
                for _, row in df.iterrows():
                    answer = _extract_boxed(row["solution"])
                    if answer is None:
                        continue
                    prompt = _math_prompt_template(row["problem"])
                    if len(prompt) > max_prompt_len * 4:  # cheap pre-filter
                        continue
                    f.write(json.dumps({"prompt": prompt, "label": answer}) + "\n")
                    written += 1
    return written


def dump_mmlu_jsonl(in_dir: Path, out_path: Path) -> int:
    """Dump MMLU questions as jsonl. Returns # examples written."""
    import pandas as pd

    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for cat_dir in sorted(p for p in in_dir.iterdir() if p.is_dir()):
            for pq in cat_dir.glob("test-*.parquet"):
                df = pd.read_parquet(pq)
                if not {"question", "choices", "answer"}.issubset(df.columns):
                    continue
                for _, row in df.iterrows():
                    q = row["question"]
                    choices = list(row["choices"])
                    if len(choices) != 4:
                        continue
                    prompt = _mmlu_format(q, choices)
                    label = idx_to_letter[int(row["answer"])]
                    fout.write(json.dumps({"prompt": prompt, "label": label}) + "\n")
                    written += 1
    return written


# ----- helpers -----------------------------------------------------

def _math_prompt_template(problem: str) -> str:
    return (
        "Solve the following math problem. Show your work and put your final "
        "answer inside \\boxed{}.\n\nProblem: " + problem
    )


def _mmlu_format(q: str, choices: List[str]) -> str:
    return (
        f"Question: {q}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        f"Answer:"
    )


def _extract_boxed(solution: str) -> str | None:
    """Return the contents of the first \\boxed{...} in solution, balancing braces."""
    m = re.search(r"\\boxed\s*\{", solution)
    if not m:
        return None
    start = m.end()
    depth = 1
    for i in range(start, len(solution)):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
            if depth == 0:
                return solution[start:i].strip()
    return None


_extract_boxed_re_fallback = re.compile(r"\\boxed\s*\{([^{}]*)\}")  # cheaper but doesn't balance


# Backward-compatible alias for code that expects a singular extractor name
_extract_boxed_first = _extract_boxed


import json  # noqa: E402  (kept after helpers since used in dump fns)
import re    # noqa: E402


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _main() -> None:
    p = ArgumentParser()
    p.add_argument("--task", required=True, choices=["math", "mmlu"])
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    out_p = Path(a.out)
    in_p = Path(a.in_dir)
    if a.task == "math":
        n = dump_math_jsonl(in_p, out_p)
    else:
        n = dump_mmlu_jsonl(in_p, out_p)
    print(f"wrote {n} records to {out_p}")


from argparse import ArgumentParser  # noqa: E402
from pathlib import Path  # noqa: E402

# Provide the sibling alias used in CLI
dump_mmlu_jsonl = dump_mmlu_jsonl


if __name__ == "__main__":
    _main()
