# Loss & Reward Reference — slime_adapter

Authoritative mapping from **every term in the optimization objective** to
**the file/lines that compute it** and **where its inputs come from**.
Every claim below is grounded in the code at its current commit.

This doc supersedes the loss/RL sections of any earlier slide deck — if a
slide and this file disagree, the file wins.

---

## 0. Big picture

We train Qwen3-30B-A3B (or any HF MoE) with **on-policy GRPO/GSPO** under
slime, plus four cache-aware modifications. The full objective is:

```
Reward (per trajectory, fed to GRPO):
    r(τ) = α_q · task(τ)                                              ← scalar correctness
         − α_c · Σ_{t,l} switch_{t,l} · n_new_{t,l}                    ← cache cost
         (NB: KL_to_teacher does NOT enter the reward — see §6.)

Advantage (slime, GRPO):
    A_t = (G_t − mean_g G) / std_g G       (group baseline, per-token broadcast)

Loss (Megatron training step):
    L  =  L_PG_token                                                  ← slime PG (token actor)
        + β · KL_advantage_shift                                       ← slime OPD-sglang KL anchor
        + λ_pg_s · L_switch_pg                                         ← OUR joint-actor PG (SwitchHead)
        + λ_h    · L_barrier                                           ← OUR hinge² overflow cap
        + λ_chunk · L_chunk_consistency                                ← OUR routing smoothness

  L_switch_pg  =  − E_i[ A_i · mean_t(Σ_l logπ(switch_{t,l}; σ_{t,l})) ]
  L_barrier    =  E[ max(0, used_t − budget)² ]
  L_chunk_*    =  routing-distribution smoothness across consecutive token chunks
```

KL appears **once**, via slime's OPD-sglang path (`--use-opd --opd-type sglang
--opd-kl-coef β`). Teacher logprobs come from the SGLang teacher server
(fetched post-rollout by `mix_generate._fetch_teacher_logprobs_full`) and
are stored on `Sample.teacher_log_probs`. slime shifts the advantage by
`−β·KL(π_S ‖ π_T)`. There is no separate KL term in the reward.

---

## 0.4. How the IS correction reaches GRPO (no metadata transfer)

Slime computes the PG ratio as ``ratio = exp(log_probs(θ) − rollout_log_probs)``
where ``rollout_log_probs`` is whatever the rollout function recorded as the
"sampling-time log-prob" on each ``Sample``. We exploit this:

  - **`mix_generate.py` writes `log p_mix(token)` to `rollout_log_probs`** at
    every step, *not* `log p_student`.
  - At training time slime's ratio becomes ``exp(logπ_θ − log p_mix) = π_θ / p_mix``,
    which is exactly the IS correction p_mix sampling needs.

So **no extra metadata transfer is required** — slime's native TIS path
handles p_mix-side IS correction for free. (Note: `Sample.metadata` is
not propagated by `slime/ray/rollout.py:_get_train_data`, so any
metadata-based IS approach would have been dead code.)

## 0.5. Sampling distribution: p_mix (MiniLLM-style)

Rollout does **not** sample from π_student. It samples from::

    p_mix(token | s_t)  =  (1 − α) · p_student(token | s_t)  +  α · p_teacher(token | s_t)

Reason: the KL-anchor (β · D_KL) only constrains in logit space and lets
the policy drift into token-space "blind spots" of the teacher to game the
reward. Sampling from p_mix forces every token to lie inside the teacher's
effective support, blocking that escape route.

The trade-off is fixed by the **importance weight** ``w_t = p_S / p_mix``:

| Item | Code | Notes |
|------|------|-------|
| Per-token mix-and-sample | `rollout/mix_generate.py:sample_p_mix` | runs against the union of student & teacher top-K |
| Token-by-token loop | `rollout/mix_generate.py:generate_one_p_mix` | two SGLang round-trips per step (parallel) |
| slime entry point | `rollout/mix_generate.py:generate_rollout` | bound via `--rollout-function-path` |
| `log p_mix(token)` storage | `Sample.rollout_log_probs` | what slime's TIS reads as "log-prob of sampling distribution" |
| IS correction | **slime's native TIS** (`slime/backends/megatron_utils/loss.py:794+`) | `ratio_t = exp(logπ_θ − log p_mix) = π_θ / p_mix` — automatic |
| `w_t = p_S/p_mix` (diagnostic only) | `Sample.metadata['importance_weights']` | not consumed by slime; logged for monitoring |

Knobs:

| flag | default | meaning |
|---|---|---|
| `--teacher-mix-alpha` | 0.5 | α — weight on teacher in p_mix |
| `--mix-top-k` | 64 | per-step vocab truncation (union of two top-Ks) |
| `--rm-url` | — | teacher SGLang base URL |

p_mix sampling is on **only** when the rollout function is set to
``slime_adapter.rollout.mix_generate:generate_rollout``. With slime's
default `sglang_rollout.generate_rollout`, IS weights are absent and
the loss patch's IS step becomes a no-op (all `w_t = 1`).

---

## 1. Reward composition (`r_t`)

| Term | Symbol | Sign | Code path |
|------|--------|------|-----------|
| Task correctness | `α_q · is_correct` | + | `slime_adapter/rollout/reward_kl.py:62-65` |
| Cache cost | `−α_c · Σ switch · n_new` | − | `slime_adapter/rollout/reward_kl.py:67-72`, `_trajectory_cache_cost` at `:90-111` |

Code reference (per-sample reward):

```python
# slime_adapter/rollout/reward_kl.py
def post_process_rewards(args, samples, **kwargs):
    α_q = float(getattr(args, "correctness_reward_alpha", 0.0))
    α_c = float(getattr(args, "cache_cost_lambda",
                         getattr(args, "budget_lambda", 0.05)))
    rewards = []
    for sample in samples:
        _stash_teacher_logprobs(sample)                      # for slime ref-KL path
        r_task = α_q * float(getattr(sample, "is_correct", 0.0))
        cache_cost = _trajectory_cache_cost(sample)          # Σ switch·n_new
        r_cache = -α_c * cache_cost
        rewards.append(r_task + r_cache)
    return rewards, list(rewards)
```

### Where the inputs come from

| Input | Source file | Source line | Origin |
|-------|-------------|-------------|--------|
| `sample.is_correct` | (rollout output) | — | optional, set by your custom RM during rollout |
| `sample.controller_records` | `slime_adapter/rollout/generate.py:32` | populated at | `RequestControllerState.serialize_for_sample_metadata` (`request_state.py:110-131`) |
| `switch`, `n_new` per (t,l) | `slime_adapter/sglang_patches/moe_select_patch.py` | populated during SGLang rollout | `RequestControllerState.record_layer_step` writes them |

> **Train-side cache (`BatchedLayerCache`)**
> Each MoE layer carries a `BatchedLayerCache` (`controller/cache_state.py`):
> a `[B, num_experts]` int64 count tensor + per-batch history deque, advanced
> sequentially over T inside `_run_switch_and_cache`
> (`megatron_hooks/moe_forward_patch.py`). Each batch row owns an
> independent rolling window, so `n_new[b, t] = |used_top2[b, t] \ cache_at(b, t)|`
> is per-(b, t) exact. The cache is reset at the top of every forward in
> `begin_controller_forward`.
>
> **DeepSets context for SwitchHead** (`expert_set_embed_dim > 0`):
> When the controller runs in DeepSets mode, the per-token forward becomes
> sequential over T so the *current* cache state and *current* router top-K
> can be encoded into fixed-dim set embeddings (`ExpertSetEncoder`,
> `controller/expert_set_encoder.py`) and fed to `SwitchHead` alongside
> `(hidden, pressure)`. SwitchHead then has direct visibility into
> "what would I have to load if I switched right now". When
> `expert_set_embed_dim == 0` (default in tests, opt-in in YAML), we keep
> the legacy vectorised SwitchHead call.
| `α_c` | YAML config | `cache_cost_lambda` (or fallback `budget_lambda`) | CLI / `args` |
| `α_q` | YAML config | `correctness_reward_alpha` | CLI / `args` |

**On-policy / off-policy:** the cache cost is computed from the **rollout**
controller records, exactly the SwitchHead/cache decisions actually used to
generate the trajectory. So the reward is on-policy by construction. GRPO's
group baseline + clipped ratio produce unbiased on-policy advantage at K=1.

---

## 2. Loss decomposition

The loss is constructed by `slime_adapter/loss/penalty_loss.py:_wrapped_policy_loss`,
which wraps slime's stock `policy_loss_function`.

```
total_loss  =  base_loss                         (slime: PG_token + ref-KL anchor)
            +  λ_pg_s · L_switch_pg              (joint actor PG over SwitchHead)
            +  λ_h    · L_barrier                (per-token hinge² on overflow)
            +  λ_chunk · L_chunk_consistency     (routing smoothness)
```

### Term 2.1 — `base_loss` (slime stock)

| Sub-term | What | Where computed | Inputs |
|----------|------|----------------|--------|
| **Token PG** `−E[ratio · clip(A_t)]` | clipped GRPO/GSPO actor on the LM head | `external/slime/slime/backends/megatron_utils/loss.py:794+` (`policy_loss_function`) inside `pg_loss = compute_policy_loss(...)` | `batch["log_probs"]` (current), `batch["rollout_log_probs"]`, `batch["advantages"]` |
| **OPD-sglang KL** `β · advantage shift` | distillation to frozen teacher | same file `:560+`, `apply_opd_kl_to_advantages` (gated by `args.use_opd`); subtracts `β·KL(π_S ‖ π_T)` from per-token advantage | `Sample.teacher_log_probs` (fetched post-rollout from SGLang teacher) |
| **Entropy** `−c_ent · H(π_θ)` | optional exploration bonus | same file, gated by `args.entropy_coef` | `log_probs`, derived `entropy` |
| **TIS / OPSM** | off-policy correction (only when K>1) | same file `:870+` | `batch["rollout_log_probs"]` |

For our **K=1 GRPO setup** (`num_update_epochs: 1` in
`configs/qwen3_30b_a3b_8gpu.yaml`): `rollout_log_probs ≈ log_probs` (closed
by RoutingReplay — see §5), so `ratio ≈ 1` and TIS is a no-op. The signal
flows through `A_t`.

### Term 2.2 — `L_switch_pg` (our joint-actor PG term)

```python
# slime_adapter/loss/penalty_loss.py: _compute_switch_pg
slp = summary.switch_logprob_per_token              # [B, T] — Σ_l logπ(switch)
advs = batch["advantages"]                          # list of 1D tensors (per-sample, response-only)

# Per-sample alignment: GRPO advantages are per-trajectory constant,
# switch_logprob is [B, T] covering full padded sequence.
a_per_sample = [a.mean() for a in advs]             # [B] scalar per sample
slp_per_sample = slp.mean(dim=-1)                   # [B] mean switch logprob per sample
L_switch_pg = -(a_per_sample * slp_per_sample).mean()
```

| Tensor | Shape | Source | Has gradient? |
|--------|-------|--------|---------------|
| `summary.switch_logprob_per_token` | `[B, T]` | `TokenBudgetState.charge_layer_with_logp` accumulates per-layer Bernoulli logπ. `slime_adapter/megatron_hooks/budget.py:83-109`. Calls into the layer wrapper at `slime_adapter/megatron_hooks/moe_forward_patch.py:218-220`. | **yes** — flows back through σ → SwitchHead linear |
| `advantages` | `list[Tensor]` | slime's `compute_advantages_and_returns` after rewards from `post_process_rewards`. Each element is 1D (response-only, variable length). Detached at the loss boundary. | no (detached) |

**Shape alignment**: slime's advantages are per-sample 1D tensors (response-only,
variable length). `switch_logprob_per_token` is `[B, T]` (full padded sequence).
We align them at the **per-sample** level: each sample's mean advantage × its
mean switch_logprob. For GRPO where advantages are per-trajectory constant,
`a.mean()` just extracts the scalar.

**Gradient path**: `L_switch_pg` → `slp` → σ → `SwitchHead.linear.{weight,bias}`.
The advantage is detached (standard PG); only the log-prob is differentiable.

**On-policy / off-policy**: this is **on-policy at K=1** (current config). At
K>1 we'd need to use `ratio_t · A_t` instead of `A_t · log π`, but `num_update_epochs: 1` so PG is the right form.

### Term 2.3 — `L_barrier` (hinge² overflow)

```python
# slime_adapter/loss/penalty_loss.py:90-92
total_credits = _resolve_total_credits(args, summary)   # 0.7 × num_moe_layers
overflow = (summary.total_used_per_token - total_credits).clamp_min(0.0)
L_barrier = (overflow * overflow).mean()
```

| Tensor | Shape | Source | Has gradient? |
|--------|-------|--------|---------------|
| `summary.total_used_per_token` | `[B, T]` | `TokenBudgetState.used_so_far`, updated by each `charge_layer_with_logp` (`budget.py:97`). STE keeps gradient flowing through `switch · n_new`. | yes (through STE) |

**Why in loss not reward**: this is a *hard constraint* — quadratic on overflow,
zero below budget. GRPO group baseline normalization would dampen the spike.
Keep it as a strong direct gradient.

### Term 2.4 — `L_chunk_consistency` (routing smoothness)

```python
L_chunk = getattr(state, "chunk_consistency_loss", None)
```

Set by the model adapter during forward (`modeling/_base.py:63`,
`modeling/qwen3_moe/adapter.py`). Differentiable through router logits.

| Field | Source | Notes |
|-------|--------|-------|
| `state.chunk_consistency_loss` | `slime_adapter/loss/chunk_consistency.py` | scalar tensor, set by adapter inside the wrapped layer forward |

### 2.5 — LoRA on expert FFN + router training

When `lora_r > 0` (YAML `lora.lora_r`), the following are applied at model
construction time by `install_controller_into_layers`:

1. **Expert LoRA** (`modeling/lora.py:apply_expert_lora`): wraps each expert's
   `linear_fc1` / `linear_fc2` with `LoRALinear(base, r, alpha)`. Custom
   wrapper is necessary because mcore's `ColumnParallelLinear` /
   `RowParallelLinear` return `(output, bias)` tuples — neither Megatron-LM
   nor HuggingFace peft provide native LoRA for mcore layers. Init: A ~
   Kaiming, B = 0 → zero delta at cold-start.

2. **Router unfreezing** (`modeling/lora.py:patch_router_gate_recompute`):
   unfreezes router params and converts to float32 (bf16 rounds gradient
   updates to zero at lr ≤ 1e-5).

3. **Parameter freezing** via slime's standard `--only-train-params-name-list`
   (regex-based, `slime/backends/megatron_utils/model_provider.py:freeze_model_params`).
   `run_train.py` auto-passes patterns `lora_ switch_head expert_set_encoder router`
   when LoRA is enabled. Only these components receive gradient; all base
   model params are frozen.

4. **Per-component learning rates** (`modeling/lora.py:collect_param_groups`):
   LoRA A/B → `lora_lr`, router → `router_lr` (float32), SwitchHead +
   ExpertSetEncoder → `controller_lr`.

#### Router gradient sources

The router gets gradient from **two paths** (neither is blocked by
RoutingReplay):

| Path | Signal | Mechanism |
|------|--------|-----------|
| **LM loss → gate weight recomputation** (primary) | Task performance | RoutingReplay replays **indices** (discrete, no grad) but **recomputes gate weights** from the current router: `probs = scores.gather(1, replayed_indices)`. `scores` = `softmax(F.linear(h, router.weight))` — fully differentiable. LM loss → `expert_output × probs` → `probs` → `scores` → `router.weight`. |
| **L_chunk_consistency** (auxiliary, λ=0.05) | Temporal smoothness | `compute_router_top_k` computes `logits = F.linear(h, router.weight)` → full `[B, T, E]` logits cached on `moe_module._slime_router_logits` → `chunk_routing_consistency_loss` uses `softmax(logits)` in a within-chunk KL → gradient flows to `router.weight`. |

Key code paths:
- `slime/utils/routing_replay.py:67-71`: `probs = scores.gather(1, top_indices)` — gate weights are differentiable
- `megatron/core/transformer/moe/moe_utils.py:719-720`: `scores = softmax(logits)` → `compute_topk(scores, ...)` — router logits enter the replay path
- `modeling/qwen3_moe/adapter.py:86-88`: `logits = F.linear(h, weight)` — separate top-k computation for chunk loss

**What the router learns**: given replayed expert indices, the router adjusts
gate **magnitudes** (relative weighting of selected experts) to minimize LM
loss. It does NOT directly learn which experts to select (selection is fixed
by replay) — but the updated router is used for selection in the **next
rollout iteration**, creating an indirect feedback loop.

---

## 3. Forward path: where every variable on `state` is born

```
TopLevelModel.forward(input_ids)
  ├─► forward-pre-hook (driver.install_forward_driver)
  │       calls begin_controller_forward(model, proxy)
  │       allocates fresh TokenBudgetState [B, T] zeros
  │
  ├─► for each MoE layer:
  │       _layer_forward_with_controller (moe_forward_patch.py:195-240)
  │           1. pressure = state.pressure_at_entry()  [B, T] detached
  │           2. σ = sigmoid(SwitchHead(hidden, pressure))
  │           3. switch = STE(σ > 0.5)
  │           4. new_top2 = adapter.compute_router_top_k(layer, hidden, k=2)
  │           5. n_new = count(new_top2 not in cache.union)
  │           6. used_top2 = where(switch, new_top2, carry-over)
  │           7. state.charge_layer_with_logp(switch, n_new, σ)
  │                  ├── used_so_far += switch * n_new           (autograd via STE)
  │                  ├── layer_costs.append(...)
  │                  └── switch_logprob_total += s·logσ + (1-s)·log(1-σ)  (autograd via σ)
  │           8. cache.push(used_top2)
  │           9. layer.controller_replay.record(...)
  │           10. forward = adapter.original_forward(hidden, forced_indices=used_top2)
  │
  └─► forward-post-hook
        - end_controller_forward(model)
        - set_last_controller_state(state)  ← TLS handoff to loss patch
```

References:
- pre-hook: `slime_adapter/megatron_hooks/driver.py:34-72`
- per-layer: `slime_adapter/megatron_hooks/moe_forward_patch.py:195-240`
- charge: `slime_adapter/megatron_hooks/budget.py:83-109`
- TLS: `slime_adapter/loss/penalty_loss.py:set_last_controller_state`

---

## 4. On-policy vs off-policy summary

| Component | On / Off-policy | Why |
|-----------|----------------|-----|
| `r_t = α_q·task − α_c·cache_cost` | **On-policy** | Computed from the actual rollout trajectory's controller_records. |
| `A_t` (GRPO group baseline) | **On-policy** | Group of N trajectories per prompt, normalized; no importance sampling. |
| Token PG `−E[ratio·clip(A)]` | **On-policy at K=1** | `ratio = exp(log_probs - rollout_log_probs) ≈ 1` at K=1. With routing replay locked, only fp16↔bf16 numerical noise remains. K>1 would require true off-policy correction (TIS); current config uses K=1. |
| Switch joint-PG `L_switch_pg` | **on-policy by construction** | We replay the same switch decisions at training time (RoutingReplay top_idx pin), so σ_train and the recorded action are consistent at K=1. Bernoulli logπ is recomputed from current σ, treating the recorded switch as the action — REINFORCE form, valid for one update step. |
| **p_mix IS weight (via `rollout_log_probs`)** | **off-policy IS, handled by slime native TIS** | Rollout draws tokens from `p_mix = (1−α)·π_S + α·π_T`, but we record `log p_mix` (not `log p_S`) into `Sample.rollout_log_probs`. slime's stock TIS then computes `ratio = exp(log π_θ − log p_mix) = π_θ / p_mix` — exactly the MiniLLM IS correction. No metadata-transfer hack needed. With α = 0, p_mix = π_S so the ratio collapses to the standard on-policy `π_θ / π_θ_old`. |
| KL_to_teacher `β·D_KL(π_θ ‖ π_T)` | **off-policy direct anchor** | Slime computes per-token `D_KL(π_θ ‖ π_ref)` from the train-side logits and the precomputed `ref_log_probs`. It's not a likelihood ratio — just an expectation taken under π_θ at the current weights, so it's "always on-policy" in the technical sense but conceptually a regularizer to a frozen teacher. |
| `L_barrier`, `L_chunk` | **direct loss** (no policy ratio) | Differentiable functions of the train-side forward; no rollout-side dependence. |

---

## 5. RoutingReplay — what it locks down

```
slime_adapter/megatron_hooks/compute_topk_patch.py
slime_adapter/sglang_patches/moe_select_patch.py
```

At rollout time, every (token, layer) records:
- `top_idx` (the experts the model chose for the actual computation)
- `switch` (cache-vs-router decision)
- `n_new` (number of new experts loaded)
- `pressure_in` (entry budget pressure)

At training time, slime's `RoutingReplay` extension (registered via
`compute_topk_patch.register_routing_replay_extensions`) overrides
Megatron's `compute_topk` to **return the recorded indices**. This makes the
train forward bit-exact with the rollout — closes the SGLang↔Megatron
numerical-drift gap, which is what `--use-routing-replay` buys us at K=1.

---

## 6. KL: where exactly does the teacher signal enter?

Two distinct paths exist in slime; **we use exactly one**:

| Path | Slime arg | Where added | Used by us? |
|------|-----------|-------------|-------------|
| β: KL as loss anchor | `--use-kl-loss --kl-loss-coef β --ref-load <teacher>` | `slime/backends/megatron_utils/loss.py:956-967` (computed from `batch["ref_log_probs"]` and added to `pg_loss`) | **no** — requires loading teacher into Megatron (2× memory) |
| **β: KL as advantage shift (OPD-sglang)** | `--use-opd --opd-type sglang --opd-kl-coef β` | `slime/backends/megatron_utils/loss.py:apply_opd_kl_to_advantages:560+` (subtracts `β·KL(stu‖teach)` from advantage using `Sample.teacher_log_probs`) | **yes** — set in YAML |

slime asserts mutual exclusivity between `kl_loss_coef > 0` and `use_opd`
(`slime/utils/arguments.py:1676`), so accidental double-counting is impossible.
Our YAML sets `use_opd: true`, `opd_kl_coef: 0.05`, and `kl_loss_coef: 0.0`.

Teacher logprobs are fetched post-rollout by
`mix_generate._fetch_teacher_logprobs_full` (one-shot HTTP call to the teacher
SGLang server) and stored on `Sample.teacher_log_probs`. This is the same
SGLang server used for p_mix sampling, so no extra infra is needed.

---

## 7. Off-policy story for K>1 (future work — currently disabled)

If we set `num_update_epochs > 1`, slime needs the importance-sampling ratio
to be principled. Three pieces become important:

1. **Token ratio** `r_t = exp(logπ_θ(token_t) − logπ_θ_old(token_t))` —
   already handled by slime's stock PG (clipped).
2. **Switch ratio** `r_t^switch = exp(Σ_l logπ_θ(switch) − Σ_l logπ_θ_old(switch))` —
   we'd need to record σ_l at rollout time (right now we record only the binary
   decision). Add to `LayerRecord.sigma` and use in the loss patch:
   ```python
   ratio_switch = (slp_new - slp_old).clamp(-clip, clip).exp()
   L_switch_pg = -(ratio_switch * adv).mean()
   ```
3. **Routing replay** is already bit-exact, so `r_t^route = 1` (no extra term).

For v1 we keep K=1 and skip (1)–(2) of the multi-epoch handling. The slides
mention K>1 as a "free bonus" — it really is once we add σ to the records.

---

## 8. Code-to-formula table (one-liner)

| Symbol in formula | Code reference |
|-------------------|----------------|
| `r(τ)` (trajectory reward) | `rollout/reward_kl.py:post_process_rewards` |
| `α_q · task(τ)` | `α_q = args.correctness_reward_alpha`, `is_correct` set by user RM |
| `α_c · Σ switch·n_new` | `rollout/reward_kl.py:trajectory_cache_cost` reads `sample.controller_records` |
| `A_t` | slime `compute_advantages_and_returns` (GRPO group baseline) |
| `−E[ratio·clip(A)]` | slime `compute_policy_loss` |
| `β·KL advantage shift` | slime OPD-sglang: `apply_opd_kl_to_advantages` using `Sample.teacher_log_probs` |
| `−E_i[A_i · mean_t(Σ_l logπ_switch)]` | `loss/penalty_loss.py:_compute_switch_pg` |
| `Σ_l logπ_switch` | `budget.py:charge_layer_with_logp` |
| `max(0, used−budget)²` | `loss/penalty_loss.py:_wrapped_policy_loss` (`L_barrier`) |
| `L_chunk_consistency` | `state.chunk_consistency_loss`, set in adapter forward |
| `cache state ω_t` | `controller/cache_state.py:LayerCache` / `BatchedLayerCache`, per-layer rolling window |
| `switch_{t,l}` (hard 0/1) | `controller/ste.py:ste_binary` (forward = `>0.5`, backward = identity through σ) |
| `σ_{t,l}` | `controller/switch_head.py:SwitchHead.forward → sigmoid` |
| `n_new_{t,l}` | `moe_forward_patch.py:_batched_n_new_loop` (train) / `BatchedLayerCache.n_new` |

---

## 9. Hyper-parameters (what the YAML knobs map to)

| YAML key | CLI arg | Used by | Symbol | Default |
|----------|---------|---------|--------|---------|
| `rollout.cache_cost_lambda` | `--cache-cost-lambda` | `post_process_rewards` | α_c | 0.0 |
| `rollout.correctness_reward_alpha` | `--correctness-reward-alpha` | `post_process_rewards` | α_q | 0.0 |
| `rollout.cache_cost_cold_start_skip` | `--cache-cost-cold-start-skip` | `trajectory_cache_cost` | — | 0 |
| `rollout.teacher_mix_alpha` | `--teacher-mix-alpha` | `generate_rollout` | α | 0.5 |
| `rollout.mix_top_k` | `--mix-top-k` | `generate_rollout` | — | 64 |
| `loss.lambda_barrier` | `--barrier-lambda` | `_wrapped_policy_loss` | λ_h | 0.5 |
| `loss.lambda_consistency` | `--consistency-lambda` | `_wrapped_policy_loss` | λ_chunk | 0.05 |
| (code default) | `args.switch_pg_lambda` | `_wrapped_policy_loss` | λ_s | 1.0 |
| `controller.budget_fraction` | `--budget-fraction` | `_resolve_total_credits` | total_credits = fraction × L | 0.7 |
| `controller.expert_set_embed_dim` | `--expert-set-embed-dim` | `wrap_moe_layer` | — | 0 (disabled) |
| `opd.opd_kl_coef` | `--opd-kl-coef` | slime OPD advantage shift | β | 0.0 |
| `model.hf_checkpoint` | `--hf-checkpoint` | slime model build | — | — |

---

## 10. Sanity checks

```python
# (a) reward path
from slime_adapter.rollout.reward_kl import trajectory_cache_cost, post_process_rewards
class S: pass
s = S(); s.controller_records = [{'switch':1,'n_new':2,'token':0,'layer':0,'used_top2':[1,2],'new_top2':[1,2],'pressure_in':0.0}]
assert trajectory_cache_cost(s) == 2.0

# (b) forward state has switch_logprob_total
from slime_adapter.megatron_hooks.budget import LayerBudgetTracker
import torch
state = LayerBudgetTracker(total_credits=4.0).begin(torch.zeros(2,4,8))
sigma = torch.full((2,4), 0.3, requires_grad=True)
switch = (sigma > 0.5).float()
state.charge_layer_with_logp(switch, torch.tensor([[2,0,1,0],[1,1,0,2]]), sigma)
assert state.summary().switch_logprob_per_token.requires_grad
```

Both pass on the current commit (verified via `pytest -k 'controller_core or integration_mock'`).
