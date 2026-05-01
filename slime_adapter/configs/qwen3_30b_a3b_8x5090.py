"""Reference config: Qwen3-MoE-30B-A3B on 8 × RTX 5090 (single node, no NVLink).

Run with::

    python configs/qwen3_30b_a3b_8x5090.py

Make sure first:
  - slime / Megatron-LM / SGLang installed (see README).
  - Teacher SGLang server running at $TEACHER_URL (see scripts/launch_teacher.sh).
  - HF & mcore checkpoints converted (slime ships a converter).

Most knobs are env-var-tunable; defaults are sensible for the reference setup.
"""

from __future__ import annotations

import os
import sys

# ----- topology / paths -----
NUM_GPUS_PER_NODE = int(os.environ.get("NUM_GPUS", 8))
MODEL_HF_PATH = os.environ.get("MODEL_HF_PATH", "/scratch/models/Qwen3-30B-A3B")
MODEL_MCORE_PATH = os.environ.get("MODEL_MCORE_PATH", "/scratch/models/Qwen3-30B-A3B_mcore")
PROMPT_JSONL = os.environ.get("PROMPT_JSONL", "/scratch/data/dapo-math-17k.jsonl")
TEACHER_URL = os.environ.get("TEACHER_URL", "http://localhost:30001")

# ----- arg groups -----

CHECKPOINT = (
    f"--hf-checkpoint {MODEL_HF_PATH} "
    f"--ref-load {MODEL_MCORE_PATH} "
)

PARALLEL = (
    "--tensor-model-parallel-size 1 "
    "--pipeline-model-parallel-size 1 "
    "--expert-model-parallel-size 8 "
    "--expert-tensor-parallel-size 1 "
    "--context-parallel-size 1 "
    "--sequence-parallel "
    "--moe-token-dispatcher-type alltoall "       # 5090 has no NVLink → don't try deepep
    "--recompute-granularity selective "
    "--use-dynamic-batch-size "
    "--max-tokens-per-gpu 4096 "
)

ROLLOUT = (
    f"--prompt-data {PROMPT_JSONL} "
    "--input-key prompt "
    "--label-key label "
    "--apply-chat-template "
    "--rollout-shuffle "
    "--rollout-batch-size 8 "
    "--n-samples-per-prompt 4 "
    "--rollout-max-response-len 1024 "
    "--rollout-temperature 1.0 "
    "--global-batch-size 32 "
    "--num-rollout 200 "
    "--use-routing-replay "
    "--rollout-num-gpus 8 --rollout-num-gpus-per-engine 4 "
    "--disable-cuda-graph "                       # M3: re-enable once mask injection is graph-friendly
    f"--rm-url {TEACHER_URL} "
    "--rm-type custom "
    "--custom-rm-path slime_adapter.rollout.reward_kl:reward_func "
    "--rollout-function-path slime_adapter.rollout.generate:generate_rollout "
)

GRPO = (
    "--advantage-estimator gspo "
    "--use-tis "
    "--num-update-epochs 1 "
    "--eps-clip 4e-4 "
    "--entropy-coef 0.0 "
    "--kl-coef 0.0 "
)

OPTIM = (
    "--optimizer adam "
    "--lr 1e-6 "                                # for router LoRA / lm_head
    "--switch-head-lr 1e-4 "                     # for the small SwitchHeads
    "--lr-decay-style constant "
    "--weight-decay 0.0 "
    "--adam-beta1 0.9 --adam-beta2 0.95 "
    "--clip-grad 1.0 "
)

CONTROLLER = (
    "--moe-arch qwen3_moe "
    "--gate-init-bias -2.0 "                     # σ ≈ 0.12 cold start
    "--use-pressure-input 1 "
    "--cache-window 16 "
    "--cache-cap 30 "
    "--budget-fraction 0.7 "                     # total_credits = 0.7 × num_moe_layers per token
    "--top-k 2 "
    "--budget-lambda 0.05 "
    "--barrier-lambda 0.5 "
    "--consistency-lambda 0.05 "
    "--chunk-size 8 "
    "--spec slime_adapter.spec:get_spec_with_controller "
)

INFRA = (
    "--actor-num-nodes 1 "
    f"--actor-num-gpus-per-node {NUM_GPUS_PER_NODE} "
    "--colocate "
    "--accumulate-allreduce-grads-in-fp32 "
    "--attention-softmax-in-fp32 "
    "--attention-backend flash "
)

SGLANG = (
    "--sglang-mem-fraction-static 0.8 "
    "--sglang-cuda-graph-max-bs 0 "  # disabled in M3
    "--sglang-max-running-requests 128 "
)


def build_command() -> str:
    return (
        "python -m slime.train "
        + CHECKPOINT
        + PARALLEL
        + ROLLOUT
        + GRPO
        + OPTIM
        + CONTROLLER
        + INFRA
        + SGLANG
    )


def main() -> None:
    cmd = build_command()
    print(cmd)
    if "--dry-run" in sys.argv:
        return
    os.execvp("bash", ["bash", "-c", cmd])


if __name__ == "__main__":
    main()
