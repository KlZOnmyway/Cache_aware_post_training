# Cache-aware MoE post-training (slime_adapter) + Original Option-Critic (rl_moe)

This repository hosts **two** generations of cache-aware MoE training code:

1. **`slime_adapter/`** — current generation. Cache-aware GRPO post-training of
   Qwen3-30B-A3B (or any HF MoE) on top of THUDM/slime + Megatron-LM + SGLang.
   This is what `scripts/quickstart.sh` launches.
2. **`train_controller_standalone.py` + `transformers_patches/`** — original
   single-host Option-Critic training of GPT-OSS-20B with mxfp4 dequantize +
   LoRA. Kept for reference; superseded by slime_adapter.

The rest of this README focuses on the slime_adapter path. The original
codebase docs are at the bottom under "Legacy: Option-Critic on GPT-OSS".

---

## Quickstart (one command)

```bash
git clone <this repo> rl_moe
cd rl_moe
bash scripts/quickstart.sh
```

That single script runs:

1. install `uv`, run `uv sync --frozen` (creates `./.venv`)
2. clone+install `slime`, `Megatron-LM`, `sglang` into `./external/`
3. start a frozen teacher SGLang server on `localhost:30001`
4. run training via `scripts/launch_train.sh configs/qwen3_30b_a3b_8gpu.yaml`

If you've already done the env / externals once, re-runs are short-circuited:

```bash
SKIP_TEACHER=1 bash scripts/quickstart.sh           # train against an existing teacher
SKIP_SETUP=1 SKIP_EXTERNALS=1 bash scripts/quickstart.sh
```

Logs land under `runs/<run_name>/` (Tensorboard) and `external/slime/...`. To
view metrics:

```bash
tensorboard --logdir runs/
```

## What this trains

A 5-modification cache-aware MoE actor — see `docs/loss_and_reward_reference.md`
for a per-term audit, or `slime_adapter/README.md` for a module summary:

1. **Per-layer rolling expert cache as MDP state ω_t**
   (`controller/cache_state.py:BatchedLayerCache`)
2. **Cache cost in the reward**, not the loss
   (`rollout/reward_kl.py:post_process_rewards`)
3. **MiniLLM-style p_mix rollout** with the `log p_mix` written to
   `rollout_log_probs` so slime's native TIS handles the IS correction
   (`rollout/mix_generate.py`)
4. **Joint actor over (token, switch_per_layer)**: extra
   `−E[A_t · Σ_l logπ(switch_{t,l})]` term in the loss
   (`loss/penalty_loss.py`)
5. **DeepSets context for SwitchHead**: cache mask + router top-K encoded
   into fixed-dim set reps and fed into σ
   (`controller/expert_set_encoder.py` + `controller/switch_head.py`)

KL-to-teacher uses slime's native OPD-sglang path (`use_opd: true,
opd_type: sglang`) — teacher logprobs come from the teacher SGLang server
that is launched by `quickstart.sh` and consumed via `Sample.teacher_log_probs`.

## File map

```
configs/qwen3_30b_a3b_8gpu.yaml   # production config (8-GPU EP=8)
scripts/
  quickstart.sh                   # one-command bootstrap (this README) ← start here
  setup_env.sh                    # uv + venv only
  install_externals.sh            # slime / Megatron-LM / sglang clone
  launch_train.sh                 # trainer launcher (called by quickstart)
  run_train.py                    # YAML → slime CLI args + run
slime_adapter/                    # the new framework (uv workspace member)
  src/slime_adapter/...           # modules: controller / loss / rollout / megatron_hooks / sglang_patches
  tests/                          # 33 unit + 5 GPU-only e2e tests
docs/
  loss_and_reward_reference.md    # canonical per-term reference (file:line)
PORT_TO_SLIME.md                  # design rationale + v4 changelog
```

## Tests

```bash
# unit + integration (no GPU, no slime needed)
uv run --frozen --no-sync python -m pytest slime_adapter/tests/ -q
# 33 passed, 5 skipped

# end-to-end against the real stack (needs GPU + ./external/* installed)
uv run --frozen --no-sync python -m pytest slime_adapter/tests/test_real_*.py -q
```

## Reading order

1. `slime_adapter/README.md` — module-level overview
2. `docs/loss_and_reward_reference.md` — every loss/reward term mapped to file:line
3. `PORT_TO_SLIME.md` — full design discussion / v1→v4 history / risks

---

# Legacy: Option-Critic on GPT-OSS

The original rl_moe scaffold (single-host LoRA-on-frozen-base + Option-Critic with
explicit V/Q/term heads) lives in:

- `train_controller_standalone.py`
- `activation_controller_trainer.py`
- `transformers_patches/`
- `launch_grid_activation.sh`

It targets gpt-oss-20b (mxfp4 storage → dequantize-on-load → bf16 + LoRA), uses
Plackett-Luce selection + Q-based ascent, and is mostly subsumed by the
slime_adapter pipeline above. Kept around for reproducing the
[paper](https://arxiv.org/abs/...) results and as a reference implementation
for the Option-Critic head architecture (`GptOssActivationController`).

## Citation

```bibtex
@article{shen2026temoe,
  title  = {Temporally Extended Mixture-of-Experts Models},
  author = {Shen, Zeyu and Henderson, Peter},
  year   = {2026},
}
```
