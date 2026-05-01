"""Real sglang ``TopK`` patch end-to-end test.

Skipped if sglang isn't importable or no CUDA. Drives
``scripts/single_gpu_real_sglang_patch.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def test_real_sglang_topk_patch():
    pytest.importorskip("sglang")
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
    except ImportError:
        pytest.skip("requires torch")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script = os.path.join(repo_root, "scripts", "single_gpu_real_sglang_patch.py")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "1")

    proc = subprocess.run(
        [sys.executable, script],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "apply_patches() ✓" in proc.stdout
    assert "with controller: TopK.forward_native returned StandardTopKOutput" in proc.stdout
    assert "state captured records: layer=3" in proc.stdout
    assert "restore_patches() ✓" in proc.stdout
