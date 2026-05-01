"""Dataset preparation utilities (MMLU / MATH → jsonl for slime --prompt-data)."""

from .prompt_loader import dump_math_jsonl, dump_mmlu_jsonl

__all__ = ["dump_math_jsonl", "dump_mmlu_jsonl"]
