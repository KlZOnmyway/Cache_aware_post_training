# slime_adapter

Cache-aware MoE post-training adapter, plugged into [THUDM/slime](https://github.com/THUDM/slime) without forking it.

We train an MoE student (reference: Qwen3-30B-A3B) under GRPO/GSPO with **five
modifications** that turn vanilla RL into cache-aware RL:

1. **Per-layer rolling expert cache as MDP state ω_t** — implemented as a
   `BatchedLayerCache` (`controller/cache_state.py`): per-layer
   `[B, num_experts]` count tensor + per-batch history deque. Each batch row
   maintains an independent 16-token rolling window so `n_new[b, t]` is
   per-(b, t) exact (no "shared union across batch" approximation).
2. **Cache cost in the reward**, not in the loss:
   `r_t = α_q · task(t) − α_c · Σ_l switch_{t,l} · n_new_{t,l}`. GRPO
   advantage attributes future cache-reuse savings back to the
   `switch=1` action that loaded the expert.
3. **MiniLLM-style p_mix rollout** to prevent reward-hacking the KL anchor
   cannot catch: `p_mix = (1−α)·π_S + α·π_T`; per-token IS weight
   `w_t = p_S/p_mix` is multiplied into the GRPO advantage by the loss patch.
4. **Joint actor over (token, switch_per_layer)**. The loss patch adds
   `−E[A_t · Σ_l logπ(switch_{t,l}; σ_{t,l})]` so SwitchHead picks up PG
   gradient through GRPO advantage in addition to its STE path.
5. **DeepSets context for SwitchHead** (`expert_set_embed_dim > 0`). Each
   layer carries an `ExpertSetEncoder` (ported from rl_moe's
   `GptOssActivationController`) that pools the current cache mask and the
   router top-K candidates into fixed-dim set embeddings, fed into
   SwitchHead alongside `(hidden, pressure)`. So σ can condition directly on
   "what would I pay if I switched right now".


The full objective:

```
r_t = α_q · task(t) − α_c · Σ_l switch_{t,l} · n_new_{t,l}              ← reward
A_t = (G_t − mean_g) / std_g                                            ← GRPO group baseline

L = −E[ratio_t · w_t · clip(A_t)]                  ← slime PG (with IS weight)
  + β · D_KL(π_θ ‖ π_T)                            ← slime KL anchor (--use-kl-loss)
  + λ_pg_s · −E[A_t · Σ_l logπ(switch_{t,l})]      ← joint actor on SwitchHead
  + λ_h · mean(max(0, used_t − budget)²)           ← hard hinge² barrier
  + λ_chunk · L_chunk_consistency                  ← routing smoothness
```

KL appears **only in the loss anchor** (slime ref-KL), never in the reward —
slime asserts at most one of `--use-kl-loss` / `--use-opd` is set, so accidental
double-counting is impossible.

For the full term-by-term audit (file:line, gradient paths, on/off-policy
classification, YAML knobs, sanity tests), see
[`../docs/loss_and_reward_reference.md`](../docs/loss_and_reward_reference.md).

## Layout

```
slime_adapter/
├── src/slime_adapter/
│   ├── controller/                  # model-agnostic core
│   │   ├── switch_head.py             SwitchHead nn.Module
│   │   ├── ste.py                     hard-forward / identity-backward
│   │   ├── cache_state.py             per-layer rolling deque
│   │   └── credits.py                 scalar credit accountant (rollout)
│   ├── modeling/
│   │   ├── _base.py                   abstract MoEModelAdapter
│   │   └── qwen3_moe/                 Qwen3-MoE concrete impl
│   ├── megatron_hooks/
│   │   ├── moe_forward_patch.py       per-layer forward wrapper
│   │   ├── compute_topk_patch.py      RoutingReplay extension
│   │   ├── budget.py                  TokenBudgetState (autograd-attached)
│   │   └── driver.py                  forward pre/post hooks
│   ├── sglang_patches/                rollout-side TopK + state mgmt
│   ├── rollout/
│   │   ├── reward_kl.py               r_traj = α_q·task − α_c·cache_cost
│   │   ├── mix_generate.py            p_mix rollout + IS weights
│   │   └── generate.py                pass-through (legacy on-policy)
│   ├── loss/
│   │   ├── penalty_loss.py            slime PG wrapper: IS+barrier+chunk+L_switch_pg
│   │   └── chunk_consistency.py
│   └── spec.py                        Megatron --spec entrypoint
├── configs/qwen3_30b_a3b_*.py
└── tests/                             21 unit + 7 p_mix + integration tests
```

## Wiring (production config)

The shipped `configs/qwen3_30b_a3b_8gpu.yaml` plugs everything in:

```yaml
rollout:
  rollout_function_path: slime_adapter.rollout.mix_generate:generate_rollout
  custom_rm_path:        slime_adapter.rollout.reward_kl:reward_func
  rm_url:                http://localhost:30001     # frozen teacher SGLang
  teacher_mix_alpha:     0.5
  mix_top_k:             64
  cache_cost_lambda:     0.05
  correctness_reward_alpha: 1.0
  use_routing_replay:    true                       # SGLang↔Megatron drift fix

loss:
  switch_pg_lambda: 1.0
  barrier_lambda:   0.5
  consistency_lambda: 0.05

grpo:
  advantage_estimator: gspo
  num_update_epochs:   1                            # K=1 strict on-policy
  use_tis:             true                         # off-policy correction (no-op at K=1)
  kl_loss_coef:        0.05                         # β · D_KL(π_θ ∥ π_ref=teacher)
  kl_coef:             0.0                          # exclusive with kl_loss_coef
```

## How to extend to a new MoE arch

The MoE-arch-specific code lives behind one abstraction
(`slime_adapter.modeling._base.MoEModelAdapter`). Everything else is model-agnostic.

To add support for, e.g., DeepSeek-MoE:

1. Implement `MoEModelAdapter` in `slime_adapter/modeling/deepseek_moe/adapter.py`:
   - `iter_moe_layers(model)` → iterable of `MoELayerHandle`
   - `compute_router_top_k(layer, hidden_states, k)` → top-K indices
   - `install_switch_head(handle, switch_head_module, attr_name)`
   - `forward_with_forced_top_indices(moe_module, hidden_states, forced_indices)`
2. Register: `register_adapter("deepseek_moe", DeepSeekMoEAdapter())`.
3. Pass `--moe-arch deepseek_moe` on the CLI.

The Qwen3-MoE reference impl in `modeling/qwen3_moe/adapter.py` is ~150 lines.

## Install

This is a uv-managed workspace member of the `rl_moe` project root. From there:

```bash
bash scripts/install_externals.sh    # clones slime / Megatron-LM / sglang into ./external
uv sync --frozen                      # installs slime_adapter + all deps
source .venv/bin/activate
```

Standalone install (controller-core tests only, no slime/sglang):

```bash
cd slime_adapter
uv pip install -e .
uv pip install pytest torch
pytest tests/test_controller_core.py tests/test_p_mix.py
```

## Testing

```bash
# unit + integration (no GPU, no real slime/sglang)
pytest slime_adapter/tests/test_controller_core.py \
       slime_adapter/tests/test_integration_mock.py \
       slime_adapter/tests/test_p_mix.py

# real-stack end-to-end (needs slime + Megatron + sglang installed)
pytest slime_adapter/tests/test_real_*  # 4 tests
```

Status: **functional**. 21/21 unit + integration tests pass; real-stack
end-to-end tested on 4-layer Qwen3-30B-A3B (truncated for CI) and 1× H100.

## Documentation

- **Production config**: `../configs/qwen3_30b_a3b_8gpu.yaml`
- **Launch script**: `../scripts/launch_train.sh`
- **Code↔formula reference**: `../docs/loss_and_reward_reference.md` (every term mapped to file:line, on/off-policy classification, YAML knobs)
- **Design rationale & history**: `../PORT_TO_SLIME.md` (long-form discussion: cache MDP, p_mix vs KL anchor, RoutingReplay, etc.)
