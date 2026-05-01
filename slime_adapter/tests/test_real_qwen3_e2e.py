"""Real Qwen3-30B-A3B (truncated) end-to-end test.

Skipped if HF cache doesn't have Qwen3-30B-A3B or no CUDA.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def test_real_qwen3_30b_a3b_e2e():
    pytest.importorskip("transformers")
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
    except ImportError:
        pytest.skip("requires torch")

    cache = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B")
    if not os.path.isdir(cache):
        pytest.skip("Qwen3-30B-A3B not in HF cache")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script = os.path.join(repo_root, "scripts", "single_gpu_real_qwen3.py")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "1")

    proc = subprocess.run(
        [sys.executable, script,
         "--num-layers", "4", "--steps", "4",
         "--batch-size", "1", "--seq-len", "8",
         "--gate-init-bias", "0.5",
         "--barrier-lambda", "1.0"],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    out = proc.stdout
    assert "adapter found 4 MoE layers" in out
    assert "H=2048 E=128 native_top_k=8" in out
    assert "OK — loss dropped" in out, "loss did not drop:\n" + out
