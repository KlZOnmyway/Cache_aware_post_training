"""Real slime end-to-end test — invokes scripts/single_gpu_real_slime.py.

Skipped if slime isn't installed or no CUDA.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def test_single_gpu_real_slime_smoke():
    """Run the single_gpu_real_slime.py script for 4 steps; assert loss drops."""
    pytest.importorskip("slime")
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
    except ImportError:
        pytest.skip("requires torch")

    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts", "single_gpu_real_slime.py"
    )
    script = os.path.abspath(script)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "1")

    proc = subprocess.run(
        [sys_executable := __import__("sys").executable, script,
         "--steps", "4",
         "--batch-size", "2",
         "--seq-len", "8",
         "--hidden-size", "32",
         "--num-layers", "3",
         "--num-experts", "8",
         "--gate-init-bias", "-1.0"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"script failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    out = proc.stdout
    assert "OK — loss dropped" in out, f"loss didn't drop:\n{out}"
    # sanity: replay_diff should hit 0 after step 0 (pure replay = bit-exact reproduction)
    assert "replay_diff=0.00e+00" in out, f"routing replay isn't bit-exact:\n{out}"
