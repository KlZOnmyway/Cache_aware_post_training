"""Production training driver — load YAML config and dispatch.

For the v1 release we support two modes:

  - ``backend: slime`` (default in production): build the slime CLI args from
    the YAML config and ``exec`` slime's ``train.py`` (assumes slime + Megatron
    are on PYTHONPATH; see scripts/launch_train.sh).

  - ``backend: in_process_smoke``: run a self-contained mini-train using
    ``scripts/single_gpu_real_qwen3.py`` semantics. Useful for sanity checks on
    a single GPU without booting Ray/SGLang.

Configs are YAML files under ``configs/``. See ``configs/qwen3_30b_a3b_8gpu.yaml``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    """Load YAML and substitute ``${var}`` references against any leaf scalar
    in the file. The returned dict preserves the YAML's nested structure
    (``cfg["model"]["hf_checkpoint"]`` etc.).
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Build a flat lookup of leaf names → values (for ${var} resolution only).
    flat = {}
    def _walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _walk(v)
            elif isinstance(v, (str, int, float, bool)):
                flat[k] = v
    _walk(cfg)

    def _subst(s):
        if not isinstance(s, str) or "${" not in s:
            return s
        out = s
        for _ in range(3):                                     # shallow refs only
            for kk, vv in flat.items():
                if isinstance(vv, (str, int, float)):
                    out = out.replace(f"${{{kk}}}", str(vv))
        return out

    def _walk_subst(d):
        for k, v in list(d.items()):
            if isinstance(v, dict):
                _walk_subst(v)
            elif isinstance(v, str):
                d[k] = _subst(v)
    _walk_subst(cfg)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backend", default=None,
                    help="override config: 'slime' (production) or 'in_process_smoke'")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    backend = args.backend or os.environ.get("CACHE_DISTILL_BACKEND", "slime")

    print(f"[run_train] config={args.config}")
    print(f"[run_train] backend={backend}")
    print(f"[run_train] run_name={cfg.get('run_name')}")
    print(f"[run_train] tensorboard={cfg.get('tensorboard_dir')}")

    if backend == "slime":
        return run_slime(cfg)
    elif backend == "in_process_smoke":
        return run_in_process_smoke(cfg)
    else:
        raise ValueError(f"unknown backend: {backend}")


# ============================================================================
# Backend 1: real slime (production)
# ============================================================================

def run_slime(cfg: dict) -> None:
    """Build slime CLI args and exec ``slime/train.py``."""
    repo = Path(__file__).parent.parent.resolve()
    slime_path = repo / "external" / "slime"
    train_py = slime_path / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"slime not installed at {slime_path}; run scripts/install_externals.sh slime")

    model = cfg["model"]
    par = cfg["parallel"]
    ctrl = cfg["controller"]
    loss = cfg["loss"]
    rollout = cfg["rollout"]
    grpo = cfg["grpo"]
    opt = cfg["optimizer"]
    infra = cfg["infra"]
    sg = cfg["sglang"]
    log = cfg["logging"]

    args_list = [
        # checkpoints
        "--hf-checkpoint", model["hf_checkpoint"],
        "--ref-load", model["ref_load"],

        # parallelism
        "--tensor-model-parallel-size", str(par["tensor_model_parallel_size"]),
        "--pipeline-model-parallel-size", str(par["pipeline_model_parallel_size"]),
        "--expert-model-parallel-size", str(par["expert_model_parallel_size"]),
        "--expert-tensor-parallel-size", str(par["expert_tensor_parallel_size"]),
        "--context-parallel-size", str(par["context_parallel_size"]),
        "--moe-token-dispatcher-type", par["moe_token_dispatcher_type"],
        "--recompute-granularity", par["recompute_granularity"],
        "--max-tokens-per-gpu", str(par["max_tokens_per_gpu"]),
        # controller
        "--moe-arch", ctrl["moe_arch"],
        "--top-k", str(ctrl["top_k"]),
        "--gate-init-bias", str(ctrl["gate_init_bias"]),
        "--cache-window", str(ctrl["cache_window"]),
        "--cache-cap", str(ctrl["cache_cap"]),
        "--budget-fraction", str(ctrl["budget_fraction"]),
        "--chunk-size", str(ctrl["chunk_size"]),
        "--spec", "slime_adapter.spec:get_spec_with_controller",
        # loss
        "--budget-lambda", str(loss["lambda_budget"]),
        "--barrier-lambda", str(loss["lambda_barrier"]),
        "--consistency-lambda", str(loss["lambda_consistency"]),
        # rollout
        "--prompt-data", rollout["prompt_data"],
        "--input-key", rollout["input_key"],
        "--label-key", rollout["label_key"],
        "--rollout-batch-size", str(rollout["rollout_batch_size"]),
        "--n-samples-per-prompt", str(rollout["n_samples_per_prompt"]),
        "--rollout-max-response-len", str(rollout["rollout_max_response_len"]),
        "--rollout-temperature", str(rollout["rollout_temperature"]),
        "--global-batch-size", str(rollout["global_batch_size"]),
        "--num-rollout", str(rollout["num_rollout"]),
        "--rollout-num-gpus", str(rollout["rollout_num_gpus"]),
        "--rollout-num-gpus-per-engine", str(rollout["rollout_num_gpus_per_engine"]),
        "--rm-url", rollout["rm_url"],
        "--rm-type", rollout["rm_type"],
        "--custom-rm-path", rollout["custom_rm_path"],
        "--rollout-function-path", rollout["rollout_function_path"],
        # GRPO
        "--advantage-estimator", grpo["advantage_estimator"],
        "--num-update-epochs", str(grpo["num_update_epochs"]),
        "--eps-clip", str(grpo["eps_clip"]),
        "--entropy-coef", str(grpo["entropy_coef"]),
        # OPD-sglang KL anchor (uses Sample.teacher_log_probs from mix_generate)
        "--kl-coef", str(grpo.get("kl_coef", 0.0)),
        "--kl-loss-coef", str(cfg.get("kl_loss_coef", 0.0)),
        # optimizer
        "--optimizer", opt["type"],
        "--lr", str(opt["lr"]),
        "--lr-decay-style", opt["lr_decay_style"],
        "--weight-decay", str(opt["weight_decay"]),
        "--adam-beta1", str(opt["adam_beta1"]),
        "--adam-beta2", str(opt["adam_beta2"]),
        "--clip-grad", str(opt["clip_grad"]),
        # infra
        "--actor-num-nodes", str(infra["num_nodes"]),
        "--actor-num-gpus-per-node", str(infra["num_gpus_per_node"]),
        # sglang
        "--sglang-mem-fraction-static", str(sg["mem_fraction_static"]),
        "--sglang-cuda-graph-max-bs", str(sg["cuda_graph_max_bs"]),
        "--sglang-max-running-requests", str(sg["max_running_requests"]),
        # slime_adapter controller knobs (passed through args namespace)
        "--lora-r", str(cfg.get("lora", {}).get("lora_r", 0)),
        "--lora-alpha", str(cfg.get("lora", {}).get("lora_alpha", 16)),
        "--expert-set-embed-dim", str(ctrl.get("expert_set_embed_dim", 0)),
        # slime_adapter rollout knobs
        "--teacher-mix-alpha", str(rollout.get("teacher_mix_alpha", 0.5)),
        "--mix-top-k", str(rollout.get("mix_top_k", 64)),
        "--cache-cost-lambda", str(rollout.get("cache_cost_lambda", 0.0)),
        "--correctness-reward-alpha", str(rollout.get("correctness_reward_alpha", 0.0)),
        "--cache-cost-cold-start-skip", str(rollout.get("cache_cost_cold_start_skip", 0)),
        # seed
        "--seed", str(cfg.get("seed", 42)),
    ]

    if par.get("sequence_parallel"): args_list.append("--sequence-parallel")
    if par.get("use_dynamic_batch_size"): args_list.append("--use-dynamic-batch-size")
    if rollout.get("apply_chat_template"): args_list.append("--apply-chat-template")
    if rollout.get("rollout_shuffle"): args_list.append("--rollout-shuffle")
    if rollout.get("use_routing_replay"): args_list.append("--use-routing-replay")
    if rollout.get("disable_cuda_graph"): args_list.append("--disable-cuda-graph")
    if grpo.get("use_tis"): args_list.append("--use-tis")
    # OPD path B (sglang teacher) — slime asserts mutual exclusivity with --use-kl-loss.
    opd_cfg = cfg.get("opd", {}) or {}
    if opd_cfg.get("use_opd"):
        args_list.append("--use-opd")
        args_list.extend(["--opd-type", str(opd_cfg.get("opd_type", "sglang"))])
        args_list.extend(["--opd-kl-coef", str(opd_cfg.get("opd_kl_coef", 0.0))])
    if infra.get("colocate"): args_list.append("--colocate")
    if infra.get("accumulate_allreduce_grads_in_fp32"):
        args_list.append("--accumulate-allreduce-grads-in-fp32")
    if infra.get("attention_softmax_in_fp32"):
        args_list.append("--attention-softmax-in-fp32")
    if infra.get("attention_backend"):
        args_list.extend(["--attention-backend", infra["attention_backend"]])
    # Parameter freezing via slime's regex mechanism (freeze everything except
    # LoRA adapters, controller heads, and router weights).
    lora_r_val = cfg.get("lora", {}).get("lora_r", 0)
    if lora_r_val > 0:
        args_list.extend([
            "--only-train-params-name-list",
            "lora_", "switch_head", "expert_set_encoder", "router",
        ])
    # Per-component LRs (passed as env vars; slime doesn't have CLI args for param-group LRs)
    for key in ("lora_lr", "router_lr", "controller_lr"):
        val = opt.get(key)
        if val:
            os.environ[f"SLIME_ADAPTER_{key.upper()}"] = str(val)

    print(f"[run_train] dispatching to slime/train.py ({len(args_list)} CLI args)")
    print(f"[run_train] cmd: python {train_py} {' '.join(shlex.quote(a) for a in args_list[:8])} ...")

    # Make sure tensorboard dir exists
    tb = cfg.get("tensorboard_dir") or cfg.get("logging", {}).get("log_dir") or "runs/default"
    Path(tb).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SLIME_ADAPTER_TB_DIR", tb)

    # exec slime trainer
    cmd = [sys.executable, str(train_py), *args_list]
    sys.exit(subprocess.call(cmd))


# ============================================================================
# Backend 2: in-process smoke (single-node, no Ray/SGLang)
# ============================================================================

def run_in_process_smoke(cfg: dict) -> None:
    """Drive ``scripts/single_gpu_real_qwen3.py`` with config-derived flags.

    Useful for verifying the controller pipeline on a single GPU without
    booting the full Ray + SGLang stack.
    """
    here = Path(__file__).parent
    smoke = here / "single_gpu_real_qwen3.py"
    ctrl = cfg["controller"]
    loss = cfg["loss"]
    rollout = cfg["rollout"]
    args_list = [
        "--num-layers", "4",
        "--steps", "8",
        "--batch-size", "1",
        "--seq-len", str(min(rollout.get("rollout_max_response_len", 64), 64)),
        "--gate-init-bias", str(ctrl["gate_init_bias"]),
        "--budget-fraction", str(ctrl["budget_fraction"]),
        "--budget-lambda", str(loss["lambda_budget"]),
        "--barrier-lambda", str(loss["lambda_barrier"]),
    ]
    cmd = [sys.executable, str(smoke), *args_list]
    print(f"[run_train] in_process_smoke: {' '.join(shlex.quote(c) for c in cmd)}")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
