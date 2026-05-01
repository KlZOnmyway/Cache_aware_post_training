# slime_adapter

Cache-aware MoE distillation adapter, designed to plug into [THUDM/slime](https://github.com/THUDM/slime) without forking it.

## What this is

We train an MoE student (e.g. Qwen3-MoE-30B-A3B) to match a frozen teacher's logits **under hardware-realistic expert-cache constraints**:

- Per-layer **rolling 16-token cache** of recently-used expert indices, capped at 30 unique experts.
- A learnable **per-layer `SwitchHead`** decides each token whether to admit fresh `router_top2` into the cache (load cost), or stick with the carried-over option (free).
- The actual MoE forward picks `top-K` from `cache ∪ {newly loaded top2}`, so cache content directly bounds expert availability.
- Budget = `0.7 × num_moe_layers` per token; `switch=1` consumes `n_new ∈ {0,1,2}` credits.

Loss (no RL, no PG, no Bernoulli):

```
L = KL(student || teacher)                                           # OPD distillation
  + λ_b · Σ_{t,l}  switch(t,l) · n_new(t,l)                           # uniform per-switch cost
  + λ_h · max(0, Σ_l switch · n_new − total_credits)²                 # token-level barrier
  + λ_c · L_chunk_routing_consistency                                  # router 时序平滑
```

`switch` is binarized via a Straight-Through Estimator (forward `1{σ>0.5}`, backward identity through σ).

See `../PORT_TO_SLIME.md` (project root) for the full design rationale.

## Layout

```
slime_adapter/
├── slime_adapter/                           # the python package (pip install -e .)
│   ├── controller/                          # model-agnostic core (no MoE knowledge)
│   │   ├── switch_head.py                   # SwitchHead nn.Module
│   │   ├── ste.py                           # STE helpers
│   │   ├── cache_state.py                   # rolling cache deque
│   │   └── credits.py                       # per-token credit accountant
│   ├── modeling/                            # MoE-architecture-specific adapters
│   │   ├── _base.py                         # abstract MoEModelAdapter
│   │   └── qwen3_moe/                       # Qwen3-MoE concrete impl
│   ├── megatron_hooks/                      # train-side patches (use modeling adapter)
│   ├── sglang_patches/                      # rollout-side patches
│   ├── rollout/                             # reward + generate
│   ├── loss/                                # slime loss monkey-patch
│   ├── data/                                # MMLU/MATH → jsonl
│   └── spec.py                              # Megatron --spec entry
├── configs/
│   └── qwen3_30b_a3b_8x5090.py
├── scripts/
└── tests/
```

## Modeling boundary

The "modeling adapter" abstraction (`slime_adapter.modeling._base.MoEModelAdapter`) is the only place that needs to know about a specific MoE architecture. Everything else (controller core, loss, rollout, sglang patch) is model-agnostic.

To support a new MoE model:

1. Subclass `MoEModelAdapter` in `slime_adapter/modeling/<your_model>/adapter.py`.
2. Implement: `iter_moe_layers`, `get_router`, `compute_router_top_indices`, `install_switch_head`, `forward_moe_with_forced_indices`.
3. Register: `register_adapter("your_model", YourModelAdapter)`.
4. Pass `--moe-arch your_model` on the CLI.

The `qwen3_moe/` subdirectory is the reference implementation.

## Install

This package is a **uv workspace member** of the `rl_moe` project. Use the
top-level workflow (one directory up):

```bash
# from rl_moe/ project root
bash scripts/setup_env.sh                # creates .venv, syncs uv.lock, full deps
bash scripts/install_externals.sh        # clones slime / Megatron-LM / sglang into ./external
source .venv/bin/activate
```

To install just `slime_adapter` standalone (without slime / Megatron / sglang),
e.g. for unit tests of the controller core:

```bash
cd slime_adapter
uv pip install -e .
uv pip install pytest torch
pytest tests/
```

For reproducible installs on a fresh server, the lockfile `rl_moe/uv.lock`
pins all transitive dep versions:

```bash
cd rl_moe && uv sync --frozen --all-extras
```

## Status

Skeleton only — implementations are stubs marked with `# TODO(slime_adapter v0.1)`. See per-file docstrings for what each piece is responsible for.
