"""Real Megatron-LM ``MoELayer`` + slime + slime_adapter end-to-end test.

Skipped if Megatron-LM isn't on PYTHONPATH or no CUDA. Drives the
``scripts/single_gpu_real_megatron.py`` smoke and asserts loss drops.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def test_real_megatron_moelayer_e2e():
    pytest.importorskip("slime")

    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
    except ImportError:
        pytest.skip("requires torch")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    mcore_path = os.path.join(repo_root, "external", "Megatron-LM")
    if not os.path.isdir(mcore_path):
        pytest.skip("Megatron-LM not cloned at external/Megatron-LM")

    script = os.path.join(repo_root, "scripts", "single_gpu_real_megatron.py")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "1")
    env["PYTHONPATH"] = (
        f"{mcore_path}:{os.path.join(repo_root, 'external', 'slime')}:"
        + env.get("PYTHONPATH", "")
    )

    proc = subprocess.run(
        [sys.executable, script,
         "--steps", "6",
         "--batch-size", "4",
         "--seq-len", "16",
         "--gate-init-bias", "0.5"],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"script failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    out = proc.stdout
    assert "TopKRouter have routing_replay ✓" in out, out
    assert "replay_diff=0.00e+00" in out, "RoutingReplay didn't reproduce bit-exact: " + out
    assert "OK — loss dropped" in out, "loss did not drop:\n" + out
