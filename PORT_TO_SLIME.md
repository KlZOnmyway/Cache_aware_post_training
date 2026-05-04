# 适配 slime 方案文档（v4：cache MDP + GRPO 联合 actor + MiniLLM p_mix + DeepSets context）

> **v4 修订（当前实现，vs v3）—— 把"带正则的监督蒸馏"纠正回真 RL**
>
> 1. **Cache state ω_t 进 MDP**：每层 rolling 16-token cache（window-only, 无 cap）作为 Markov 状态。switch action 物理改变 ω_{t+1}，进而影响未来 reward。
> 2. **Cache cost 进 REWARD（不再进 loss）**：`r_t = α_q·task − α_c·Σ_l switch_{t,l}·n_new_{t,l}`。GRPO advantage A_t 把"future expert reuse 节省的成本"反向传回 t 时刻的 switch action。
> 3. **GRPO 联合 actor**：策略 `π_θ(token, switch_1..L | s_t)`。Loss patch 多一项 `L_switch_pg = −E[A_t · Σ_l logπ(switch_{t,l})]`，给 SwitchHead 真正的 PG 梯度。STE 仍用于 forward 的硬决策。
> 4. **MiniLLM p_mix rollout**：student 从 `(1−α)·π_S + α·π_T` 采样；per-token IS 权重 `w_t = p_S/p_mix` 在 loss 端乘进 advantage。是 KL anchor 防不住的 token-空间 reward hacking 的解。
> 5. **SwitchHead + DeepSets context**：每层挂 `ExpertSetEncoder`（`Embedding(E, d) → φ MLP → mean-pool`），把当前 cache mask + router top-K 编成 set rep 喂给 SwitchHead，让它直接看到"现在切要付几张票"。`expert_set_embed_dim>0` 启用。
> 6. **`BatchedLayerCache`**：cache 改成 `[B, num_experts]` 计数张量 + per-batch history deque，每个 batch 行独立。`n_new` 在每个 (b,t) 上精确，不再"全 batch 共享 union"近似。
> 7. **KL anchor 在 advantage 里（OPD-sglang, 不进 reward）**：用 slime 原生 `--use-opd --opd-type sglang --opd-kl-coef β`。teacher logprobs 从 SGLang server 获取，β·KL 从 advantage 中扣减。不需额外加载 teacher 到 Megatron。
> 8. **RoutingReplay 留下来**：消除 SGLang ↔ Megatron 之间 router top-K 数值翻转。K=1 单轮也必须开。
>
> 完整 loss + reward + 文件:行号见 [`docs/loss_and_reward_reference.md`](docs/loss_and_reward_reference.md)。`PORT_TO_SLIME.md` 是高层架构 / 文件分工 / 里程碑 / 风险。
>
> ---
>
> v3（已弃用）：cache cost 还在 loss、纯 supervised；`switch·n_new` 改 layer-uniform；新增 hinge² barrier 和 chunk consistency；pressure 改作 SwitchHead 输入；可学 `per_layer_bias`。
>
> v2（已弃用）：把 Option-Critic（V/Q/PL/GAE）整套换成 STE + token-level budget。纯蒸馏。
>
> v1（已弃用）：直接搬 rl_moe 的 Option-Critic。

---

## 0. TL;DR

- 把 slime 当三方库装（`pip install -e .` 配合 `bash scripts/install_externals.sh`），所有适配代码集中在 `slime_adapter/` 包里。**不 fork slime，不改 slime 源码**——所有 hook 走 monkey-patch + plugin。
- Megatron-LM 和 SGLang 各打一套运行期 patch；slime 自带 RoutingReplay 扩展。
- 5090 / H100 都行：8-GPU EP=8 + sequence parallel + bf16，**不量化、不 LoRA**，每卡 ~5 GB。
- 算法：**GRPO actor**（group baseline 替代 V/Q），联合 actor 同时驱动 LM head 和 per-layer SwitchHead；**MiniLLM p_mix** 防 reward hacking；**RoutingReplay** 消除 SGLang↔Megatron 数值漂移。
- Cache state ω_t 是真正的 MDP 状态：switch action 改它，它改 future reward；折扣回报让 PG 学到"什么时候花什么时候省"。
- 完整 loss / reward / 文件:行号见 `docs/loss_and_reward_reference.md`。

## 1. 算法形式（实现，便于对照代码）

> 这一节是**当前实现的算法形式**——和 `slime_adapter/src/slime_adapter/` 1:1 对应。
> 完整的 loss + reward + 文件:行号请直接看
> [`docs/loss_and_reward_reference.md`](docs/loss_and_reward_reference.md)。

### 1.1 单层 forward（每个 MoE block 内部）

```
# 进入第 l 层时已知：
#   hidden_t,l ∈ [B, T, H]                 (前置 attention/normalize 输出)
#   pressure_t = used_credits_t / total    (forward-only 标量, detach)

# DeepSets context (可选, 若 expert_set_embed_dim>0)：
cache_mask     ∈ [B, num_experts] bool             # ω_t 的 union mask
new_top_k      = router_top_k(hidden_t,l, k=2)     # router 偏好
cache_set_rep  = ExpertSetEncoder.encode_mask(cache_mask)   # [B, set_dim]
top_k_set_rep  = ExpertSetEncoder.encode_indices(new_top_k) # [B, set_dim]

# SwitchHead：决定该不该切 (forward 硬决策, backward 通过 σ 走 PG)
σ_l        = sigmoid(switch_head(hidden_t,l, pressure_t,
                                  cache_set_rep, top_k_set_rep))
switch_l   = STE(σ_l)                              # forward I[σ>0.5], backward = σ

# 拿决策施加在 cache 状态机上
used_top_k = switch_l ? new_top_k : current_top_k
n_new      = |used_top_k \ ω_t|                    # 0..k

# 推 cache（per-batch 独立, BatchedLayerCache, window-only eviction）
ω_{t+1}    = window-rolling(ω_t ∪ used_top_k, window=16)

# 累计预算账（per token, 跨层加和, layer 0 进入时 reset）
used_credits_{t,l+1} = used_credits_{t,l} + switch_l · n_new_{t,l}
```

### 1.2 Per-trajectory reward（rollout 末尾算）

```
r(τ) = α_q · is_correct(τ)                                     # task reward
     − α_c · Σ_t Σ_l  switch_{t,l} · n_new_{t,l}                # cache 成本
```

KL_to_teacher **不进 reward**——它通过 slime 的 `--use-kl-loss + --ref-load <teacher>`
在 loss 端作 anchor。

### 1.3 Per-step training loss（当前实现）

```
L  =  L_PG_token                                                ← slime PG（token 维 actor）
    + β · KL advantage shift                                     ← slime OPD-sglang (--use-opd)
    + λ_pg_s · −E_i[ A_i · mean_t(Σ_l logπ(switch_{t,l})) ]     ← 联合 actor PG（SwitchHead）
    + λ_h    · mean( max(0, used_t − total_credits)² )           ← 硬 hinge² barrier
    + λ_chunk · L_chunk_consistency                              ← router 时序平滑

A_t  = (R_τ − mean_g R) / std_g R                                ← GRPO group baseline
ratio_t = exp( logπ_θ(token_t) − log p_mix(token_t) )             ← TIS IS 校正（slime 原生）
```

`log p_mix` 存入 `Sample.rollout_log_probs`，slime 的 TIS 自动计算
`ratio = π_θ / p_mix`，即 IS 校正。不需要通过 metadata 传 IS 权重。

### 1.4 关键梯度路径

| 模块 | 梯度来源 | 链路 |
|---|---|---|
| `lm_head / Megatron actor` | `L_PG_token` (slime) | `−ratio·clip(A_t·w_t)` → log p_token |
| `router (per-layer)` | `L_KL_anchor` + `L_chunk_consistency` | KL 锚 + router 概率平滑（不被 RoutingReplay 阻断的那条 hidden 链） |
| `SwitchHead.linear` | `L_switch_pg` | `−A_t · log π(switch_l)` 通过 σ_l 反传 |
| `SwitchHead.linear` (二次通道) | `L_KL_anchor` 通过 STE | switch 在 forward 选了 used_top_k，影响 hidden chain，KL 梯度反穿 STE 回到 σ_l |
| `SwitchHead.linear` (三次通道) | `L_barrier` | 超预算时 hinge² 通过 used 累计回到 σ_l (cost 是 `switch·n_new` 含 σ STE) |
| `ExpertSetEncoder` (DeepSets) | 全部包含 cache_set_rep / top_k_set_rep 输入的项 | 通过 SwitchHead 第一层 Linear 反传到 φ MLP 和 expert embedding |
| `BatchedLayerCache` | **不可微**：n_new 是整数计数，没有梯度路径 | 仅前向账本 |

### 1.5 与 v3 的关键区别

| | v3（旧）| v4（当前）|
|---|---|---|
| 范式 | 监督蒸馏 + 正则 | RL（GRPO actor + group baseline） |
| Cache cost 位置 | loss 项 `λ_b · Σ switch·n_new` | reward 项 `−α_c · Σ switch·n_new` |
| SwitchHead 梯度来源 | STE 通过 KL 一条 | PG（A_t · logπ） + STE × KL + barrier 三条 |
| Cache 实现 | 单 trajectory 的 Python deque, batch 共享 union | `BatchedLayerCache`（[B, E] 张量, per-batch 独立, window-only） |
| SwitchHead 输入 | (h, pressure) | (h, pressure) + DeepSets(cache_mask, top_k) |
| Rollout | 纯 student | MiniLLM p_mix(α=0.3) + log p_mix → TIS IS 校正 |
| KL 信号 | reward + loss 都有（双计） | OPD-sglang advantage shift（slime --use-opd）|
| Cache 与训练 forward 一致性 | 不严格 | RoutingReplay 强制 SGLang↔Megatron top-K 一致 |

---


## 1. 架构总览

```
┌──────────────────────┐    teacher logp    ┌────────────────────┐
│ teacher SGLang       │ ──────────────►    │ Sample.teacher_lp  │
│ (gpt-oss frozen,     │                    │ (per token)        │
│  no controller)      │                    └────────────────────┘
└──────────────────────┘                              ▲
                                                      │
┌──────────────────────┐    student tokens   ┌────────────────────┐
│ student SGLang       │ ──────────────►    │ Sample.tokens,     │
│ (with controller +   │   record:          │   .top2_per_layer, │
│  cache state machine)│   used_top2,       │   .switch_per_layer│
│                      │   switch, n_new    │   .cache_state     │
└──────────────────────┘                    └────────────────────┘
                                                      │
                                  ┌───────────────────▼─────────────────┐
                                  │ Megatron training (replay forward)  │
                                  │  - RoutingReplay -> top2 from record│
                                  │  - cache state from record (固定)   │
                                  │  - σ_l = sigmoid(switch_head(h))   │
                                  │  - STE forward, gradient backward   │
                                  │  - L = KL + λ · Σ_l penalty         │
                                  └─────────────────────────────────────┘
```

- **生成** 与 **训练** 物理拆开（slime 的标准模式）：SGLang 服务负责吞吐，Megatron 负责梯度。
- **Teacher** 是另一个 SGLang server（无 controller、router 原始权重），rollout 时一次性提供 token-level logp。
- **routing replay**（slime 自带 `--use-routing-replay`）保证 Megatron 这边训练 forward 走的是 rollout 时被记录的 top2 expert 路径。

---

## 2. slime 当作三方库 + plugin（不 fork）

slime 已经把所有自定义点暴露为 CLI 字符串路径：

| 扩展点 | slime CLI | 我们的实现 |
|---|---|---|
| Megatron layer spec | `--spec` | `slime_adapter.spec:get_gpt_oss_with_gate_spec` |
| Rollout 主循环 | `--rollout-function-path` | 默认 `slime.rollout.sglang_rollout.generate_rollout`（基本不改） |
| 自定义生成（控制每 token 路由） | 通过 SGLang patch 实现，不动 slime | — |
| Reward 函数 | `--custom-rm-path` | `slime_adapter.rollout.reward_kl:reward_func`（OPD 风格） |
| Routing replay | `--use-routing-replay`（slime 内建） | top_k=2 直接复用，不需扩展 |
| 训练 loss 注入 budget penalty | import-time monkey-patch | `slime_adapter.loss.penalty_patch:apply()` |

**zero fork**：slime 源码一行不动；Megatron-LM 和 SGLang 各一个本地 patch（CI 自动 apply）。

---

## 2. 仓库目录

```text
external/
  slime/                 # git clone THUDM/slime; pip install -e .
  Megatron-LM/           # nvidia 源 + slime 的 docker/patch/latest/megatron.patch
                         #              + slime_adapter 的 controller patch (新增)
  sglang/                # sglang 源 + slime_adapter 的 cache+gate patch (新增)

rl_moe/                  # 当前仓库
  PORT_TO_SLIME.md       # 本文档
  slime_adapter/                                  # 新增
    __init__.py
    spec.py                                       # Megatron layer spec 注入
    controller/
      __init__.py
      switch_head.py                              # 唯一可学模块: Linear(H, 1) per layer
      ste.py                                      # straight-through helper
      cache_state.py                              # per-(layer,request) cache deque & union
      credits.py                                  # per-token credits 累加器
    megatron_hooks/
      __init__.py
      moe_forward_patch.py                        # 在 MoE forward 前插 switch + cache
      compute_topk_patch.py                       # 顺手强制 top_k=2，对接 RoutingReplay
    sglang_patches/
      __init__.py
      moe_select_patch.py                         # SGLang select_experts 注入相同逻辑
      request_state.py                            # 每 request 的 cache + budget 状态
    rollout/
      __init__.py
      reward_kl.py                                # OPD reward, 调 teacher SGLang
      generate.py                                 # 仅在需要透传额外 metadata 时定义
    loss/
      __init__.py
      penalty_loss.py                             # monkey-patch slime loss 加 λ·Σ penalty
    data/
      math_loader.py                              # 把 MATH/MMLU 转 jsonl 给 slime
    configs/
      gpt_oss_20b_8x5090.py
      gpt_oss_20b_debug.py
```

---

## 3. 复用 vs 新写清单

### 直接复用（不动）

| 来源 | 用途 |
|---|---|
| `slime_plugins/mbridge/gpt_oss.py` | gpt-oss-20b 的 HF↔Megatron 权重转换桥 |
| `slime_plugins/models/gpt_oss.py` | Megatron transformer block spec for gpt-oss |
| `slime/utils/routing_replay.py` | record / replay top2 expert ids；我们设 `top_k=2` 即可，不扩展 |
| `slime/rollout/on_policy_distillation.py` | teacher logprob 拉取范式，我们 fork 一份做 KL reward |
| `slime/backends/megatron_utils/sglang.py` + `slime/ray/*` | 训-推权重同步、rollout/train 资源调度 |
| `slime/backends/megatron_utils/loss.py:loss_function` | 现有 PG / KL loss 框架，我们 monkey-patch 加 penalty 项 |

### 来自现 rl_moe 仓库可整块搬迁

| 现仓库代码 | 去向 | 改动 |
|---|---|---|
| `KLReward.compute_kl_divergence` 的 KL 数学 | `slime_adapter/rollout/reward_kl.py` | teacher logits 改从 SGLang server 取 |
| `eval/eval_math.py:is_equiv` | `slime_adapter/rollout/reward_kl.py` | correctness reward 加成（可选） |
| `collect_math_prompts / collect_mmlu_prompts` | `slime_adapter/data/math_loader.py` | 提前 dump 成 jsonl 给 slime `--prompt-data` |

### 全部丢弃（v1 列入复用、v2 不再需要）

| 现仓库代码 | 原因 |
|---|---|
| `GptOssActivationController` 的 selection_head / DeepSets / PL sampler | 没有 selection 动作了 |
| `GptOssJointOptionController` 整个 | joint option 概念不复存在 |
| `_compute_single_rollout_loss` / `_compute_intra_option_loss` 的 GAE / advantage / Q targets | 直接 supervised loss，不用 |
| Plackett-Luce log_prob 系列函数 | 同上 |
| `_swap_router_weights` / 双 KV cache teacher mix / `_generate_with_teacher_mix` | OPD 模式下 teacher 是独立 server |
| `peft.LoraConfig` + `Mxfp4Config(dequantize=True)` | EP=8 不需要 LoRA；权重直接 bf16 |
| `freeze_non_controller` + 手工 `torch.distributed.all_reduce(p.grad)` | Megatron 自管 |
| `controller_trainer.py` 整个 | 不复存在 |
| `activation_controller_trainer.py` 整个 | 同上 |

---

## 3. 算法 → 代码：每个文件该写什么

### 3.1 `slime_adapter/controller/switch_head.py`

v3：switch_head 接受额外的 `pressure` 输入特征（当前层进入时的预算压力），让模型显式条件化于 budget 状态。每层独立的 `per_layer_bias` 是可学的层级先验。

```python
class SwitchHead(nn.Module):
    """每层一个 Linear((H+1) → 1) + 可学层级 bias。
    输入: hidden state + budget pressure scalar (forward-only).
    输出: switch logit, 经 sigmoid + STE 二值化."""
    def __init__(self, hidden_size, init_bias: float = -2.0):
        super().__init__()
        # +1 因为还要拼一个 pressure scalar
        self.linear = nn.Linear(hidden_size + 1, 1)
        nn.init.zeros_(self.linear.weight)
        # 把 bias 初始化在最后一维，让初始 σ 接近 sigmoid(init_bias)
        nn.init.constant_(self.linear.bias, init_bias)

    def forward(self, hidden, pressure_scalar=None):
        # hidden: [..., H], pressure_scalar: [...] or None (退化到 v2 仅 hidden)
        if pressure_scalar is not None:
            x = torch.cat([hidden, pressure_scalar.unsqueeze(-1)], dim=-1)
        else:
            x = hidden
        return self.linear(x).squeeze(-1)
```

设计要点：

- `use_pressure_input=True`：把 `pressure` 拼到 hidden 后做 switch_head 的输入，**仅 forward**。让 σ 决策时显式看到当前 token 已用预算，模型可学到"剩余多敢切 / 剩余少要保守"的条件策略。
- `pressure` 在 forward 时已 detach，**梯度不会从后续层错误地反传到前层 σ**——梯度路径只走 layer 自己的 KL chain + uniform cost + barrier。
- `per_layer_bias`（实现里就是 `Linear` 的 bias）作为可学层级先验，让模型自适应分化"我这层一般敢切 / 一般保守"，把**层级 importance** 从 cost 公式挪进网络。
- `init_bias = -2.0` 让起步 σ ≈ 0.12，避免冷启动期 switch 全开把 budget 一次性烧光。

参考 v2（不带 pressure 输入，仅作历史对比）：

```python
import torch.nn as nn

class _OLD_SwitchHead_v2(nn.Module):
    """每层一个 Linear(hidden_size, 1)。整个模型的可学控制器参数总量。"""
    def __init__(self, hidden_size: int, init_bias: float = -2.0):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, init_bias)  # 初始 σ ≈ 0.12

    def forward(self, hidden):  # hidden: [B, T, H] or [B*T, H]
        return self.linear(hidden).squeeze(-1)  # logit
```

> 说明 ：参考实现里把 v3 的 `(hidden_size + 1, 1)` 输入改回 `(hidden_size, 1)`、去掉拼接，就是 v2 行为，留作 ablation 用。

### 3.2 `slime_adapter/controller/ste.py`

```python
def ste_binary(sigma):
    """Straight-Through Estimator for sigmoid → binary.
    Forward: hard threshold at 0.5
    Backward: gradient of identity (∂switch/∂σ = 1)
    """
    hard = (sigma > 0.5).to(sigma.dtype)
    return hard.detach() + sigma - sigma.detach()
```

可选：温度 / 噪声混入，前期训练加点扰动避免 σ 全部饱和：

```python
def ste_binary_noisy(sigma, noise_std: float = 0.0):
    if noise_std > 0:
        sigma = (sigma + torch.randn_like(sigma) * noise_std).clamp(0, 1)
    return ste_binary(sigma)
```

### 3.3 `slime_adapter/controller/cache_state.py`

```python
from collections import deque

class LayerCache:
    """16-token rolling deque + union set, per layer per request."""
    WINDOW = 16
    CAP = 30

    def __init__(self):
        self.deque: deque[tuple[int, int]] = deque(maxlen=self.WINDOW)  # (e0, e1)
        self.union_count: dict[int, int] = {}  # expert_id -> #occurrences

    def n_new(self, top2: tuple[int, int]) -> int:
        return sum(1 for e in top2 if self.union_count.get(e, 0) == 0)

    def push(self, top2: tuple[int, int]):
        if len(self.deque) == self.WINDOW:
            old = self.deque[0]
            for e in old:
                self.union_count[e] -= 1
                if self.union_count[e] == 0:
                    del self.union_count[e]
        self.deque.append(top2)
        for e in top2:
            self.union_count[e] = self.union_count.get(e, 0) + 1
        # CAP 30 几乎永远不会触发；触发时强制 evict 最老 entry 的某个 expert
        # (生产代码加防御性 fallback)
```

`union_count` 是引用计数，使得 evict 一个 token 的 entry 时只丢"已经没人用了"的 expert id。

### 4.4 `slime_adapter/controller/credits.py`

```python
class CreditsTracker:
    """Per-token, per-rollout: 在 layer 0 重置，按 layer 顺序累加。"""

    def __init__(self, total: float):
        self.total = total
        self.used_so_far = 0.0  # 训练时用 σ × n_new；推理时用 hard switch × n_new

    def reset_for_token(self):
        self.used_so_far = 0.0

    def step(self, switch_signal, n_new: int) -> float:
        """switch_signal: torch.Tensor (forward 用 hard, backward 走 σ via STE).
        Returns penalty AFTER this layer = used / total."""
        self.used_so_far = self.used_so_far + switch_signal * n_new
        # 返回的不是 cost 项，而是 pressure 标量（switch_head 的输入特征）
        # 注意：这里返回的是"进入下一层时的 pressure"
        return (self.used_so_far / self.total).detach()
```

`step()` 返回的是 **pressure**（forward-only 输入特征），不是直接的 cost——v3 把 pressure 从 cost 公式里拿掉了。每层的 cost 由外层 loss 用 `switch_l · n_new_l` 直接累加，与 pressure 无关。

每个 token 进入新 forward 时调 `reset_for_token()`。

### 4.5 `slime_adapter/megatron_hooks/moe_forward_patch.py`

注入到 Megatron 的 `MoELayer.forward` 前。三件事：
1. 取 entry-time `pressure_l`，喂给 switch_head 与 hidden 一起算 σ_l。
2. 用 STE 决定本层是否走 new top2。
3. 把本层的 (`switch_l · n_new_l`) 推到 `layer_local_costs` collector；累计到 `total_used_per_token` 用于 barrier。

```python
def patched_moe_forward(self, hidden_states, *args, **kwargs):
    # 1. 路由前的 controller 决策
    pressure_entry = (credits_state.used_so_far / credits_state.total).detach()  # forward-only feature
    sigma   = sigmoid(self.switch_head(hidden_states, pressure_entry)).clamp(1e-5, 1 - 1e-5)
    hard    = (sigma > 0.5).to(sigma.dtype)
    switch  = hard.detach() + sigma - sigma.detach()                              # STE

    # 2. router top2 (确定性 argmax)
    new_top2 = self._router_topk(hidden_states, k=2)

    # 3. 取 cache & 计算 n_new
    cache, credits_state = self._get_per_request_cache(hidden_states)
    n_new = compute_n_new(new_top2, cache)            # forward 中算，不进梯度

    # 4. used_top2: switch=1 用 new_top2; switch=0 用 cache 里 carry over 的 current_top2
    #    (实际通过 RoutingReplay 注入，见 §4.6)

    # 5. credits 状态机 + 本层 cost
    pressure_entry = credits_state.used_so_far / credits_state.total   # 进入本层时的 pressure
    pressure_entry_detached = pressure_entry.detach()                  # 仅作 switch_head 输入
    # 注意: σ 已经在 step 1 用 pressure_entry_detached 算过了
    credits_state.used_so_far = credits_state.used_so_far + switch * n_new
    layer_local_cost = switch * n_new          # uniform per-switch cost, 可微 via STE
    self._record_layer_local_cost(layer_local_cost=layer_local_cost,
                                   total_used_per_token=credits_state.used_so_far)

    # 6. 原 MoE forward
    return original_moe_forward(self, hidden_states, used_top2=used_top2, *args, **kwargs)
```

`self._record_layer_cost(...)` 把每层每 token 的 `layer_local_cost = switch · n_new` 推到 `model_output.layer_local_costs[l]`，把累计 `total_used_per_token` 推到 `model_output.budget_overflow_per_token`（在 token 末尾算 `max(0, total - 0.7L)`）。loss function（§3.6）一次性求 mean 求 hinge²。

### 3.4 `slime_adapter/megatron_hooks/compute_topk_patch.py`

slime `RoutingReplay` 已经能接管 `compute_topk`：

```python
from slime.utils.routing_replay import get_routing_replay_compute_topk
import megatron.core.transformer.moe.router as _router
_router.compute_topk = get_routing_replay_compute_topk(_router.compute_topk)
```

放在 `slime_adapter/__init__.py` 的 import 时即生效，需要把 top_k 强制为 2（gpt-oss 默认 4）。这点通过 `slime_plugins/mbridge/gpt_oss.py` 里 `moe_router_topk: 2` 配置即可，不需要再改 patch。

### 3.5 `slime_adapter/sglang_patches/moe_select.py`

SGLang 端镜像同样的 controller 逻辑——但**完全推理路径**（`@torch.no_grad()`，hard switch，update cache，记账）：

```python
import sglang.srt.layers.moe.topk as _topk
_orig = _topk.select_experts

def patched_select(router_logits, top_k, *args, **kwargs):
    state = current_request_state.get()  # context-local
    layer_idx = state.current_layer_idx
    h = state.last_hidden  # 由上层 hook 注入

    # entry-time budget pressure as input feature (no_grad on rollout side)
    pressure = state.credits_used / state.credits_total
    sigma = torch.sigmoid(state.switch_head[layer_idx](h, pressure))
    switch = (sigma > 0.5)

    new_top2 = _orig(router_logits, top_k=2, ...)  # router argmax

    cache = state.cache[layer_idx]
    n_new = cache.n_new(new_top2)
    used_top2 = new_top2 if switch else state.current_top2[layer_idx]
    cache.push(used_top2)

    state.current_top2[layer_idx] = used_top2
    if switch:
        state.credits_used += n_new                      # token 内累加; 跨 token reset
    state.record(layer_idx, switch=int(switch), used_top2=used_top2,
                 new_top2=new_top2, n_new=n_new, pressure_at_entry=pressure)

    return used_top2

def on_token_boundary(state):
    """每个新 token 起始时 reset budget. 由 sglang generate loop 调用."""
    state.credits_used = 0.0
```

这是整个适配里**唯一需要改 SGLang 源码**的地方。CUDA graph capture 与可变 mask 的兼容性是已知风险，一期建议 `--disable-cuda-graph`，二期再做 graph-friendly 化。

### 3.5 `slime_adapter/rollout/reward_kl.py`

OPD-flavoured reward + per-trajectory cache cost. Does **not** put
KL_to_teacher into the reward — that path goes through slime's
`--use-kl-loss + --ref-load <teacher>` anchor at the loss level, so the role
of `reward_kl` is:

1. ``reward_func(args, sample, ...)`` (async) — query the frozen teacher
   SGLang for per-token logprobs and stash on ``sample._raw_reward_response``.
   ``post_process_rewards`` parses it into ``sample.teacher_log_probs`` for
   slime's ref-KL machinery downstream.

2. ``post_process_rewards(args, samples, ...)`` — compose the scalar GRPO
   reward:

   ```python
   r_traj  =  α_q * sample.is_correct
            − α_c * trajectory_cache_cost(sample)         # Σ_t Σ_l switch·n_new
   ```

   ``trajectory_cache_cost`` reads ``sample.controller_records`` (every
   (token, layer) records ``switch``, ``n_new``, ``used_top2``, ``new_top2``,
   ``pressure_in``).

Knobs on ``args``:

  - ``correctness_reward_alpha`` (α_q) — task-correctness weight; default 0.0
  - ``cache_cost_lambda`` (α_c) — falls back to ``budget_lambda``; default 0.05

### 3.5.b `slime_adapter/rollout/mix_generate.py`

MiniLLM-style p_mix rollout. Wired via
`--rollout-function-path slime_adapter.rollout.mix_generate:generate_rollout`.

For each trajectory, token-by-token loop:

1. ``await asyncio.gather`` two SGLang calls (student / teacher) for the
   top-K next-token logprobs at the current prefix.
2. Renormalize within the union of the two top-K supports.
3. ``p_mix = (1 − α) · p_S + α · p_T``; sample one token.
4. Record per-step ``w_t = p_S(token) / p_mix(token)`` and
   ``log p_S(token)`` onto the Sample's ``metadata`` /
   ``rollout_log_probs``.

Two SGLang calls per generated token in parallel via ``asyncio.gather``;
throughput comes from running many trajectories concurrently and letting
SGLang's continuous batching handle the inter-trajectory parallelism.

Knobs on ``args``:

  - ``teacher_mix_alpha`` (default 0.5)
  - ``mix_top_k`` (default 64)
  - ``rm_url`` — teacher SGLang base URL

### 3.6 `slime_adapter/loss/penalty_loss.py`

Wraps slime's stock ``policy_loss_function`` to (a) apply per-token p_mix
importance weights to the advantages, and (b) add the cache-aware aux
terms. Pseudo-code:

```python
def _wrapped_policy_loss(args, batch, logits, sum_of_sample_mean):
    # (1) IS-weight rescale: advantages[i] *= w_t[i]   (no-op without p_mix)
    _apply_importance_weights_inplace(args, batch)

    # (2) slime's stock PG + KL anchor + entropy + TIS — sees rescaled advantages
    base_loss, base_metrics = _orig_policy_loss(args, batch, logits, ...)

    # (3) Pull controller readout from TLS (set by the forward driver)
    state = state_or_none()
    if state is None:
        return base_loss, base_metrics
    summary = state.summary()        # BudgetReadout

    # Joint-actor PG term — SwitchHead's PG signal
    advantages = _extract_advantages(batch).detach()           # [N]
    L_switch_pg = -(advantages * summary.switch_logprob_per_token).mean()

    # Hard barrier on per-token credits overflow
    overflow  = (summary.total_used_per_token - total_credits).clamp_min(0.0)
    L_barrier = (overflow * overflow).mean()

    # Routing smoothness (set by the model adapter)
    L_chunk   = getattr(state, "chunk_consistency_loss", 0.0)

    aux = (args.switch_pg_lambda * L_switch_pg
         + args.barrier_lambda    * L_barrier
         + args.consistency_lambda * L_chunk)
    return base_loss + aux, {**base_metrics, "loss/switch_pg": ..., ...}
```

Important: the **token-side PG** term and **KL_to_teacher anchor** live
inside `base_loss` (slime's stock path), they are **not** re-implemented
here. The wrapper only adds the *new* signals — joint-actor switch PG,
hinge² barrier, chunk consistency — and applies the IS rescale.

Hyperparameters (all on `args`):

  - `switch_pg_lambda` (λ_pg_s)
  - `barrier_lambda`   (λ_h)
  - `consistency_lambda` (λ_chunk)
  - `kl_loss_coef`     (β; slime native, controls the KL anchor)
  - `use_tis`, `eps_clip`, `eps_clip_high`, … (slime native, GRPO/GSPO)

Wired in via:

```python
# scripts/run_train.py
import slime_adapter.loss.penalty_loss as _pl
_pl.apply_patch()
```


### 3.7 `slime_adapter/spec.py`

```python
from slime_plugins.models.gpt_oss import get_gpt_oss_spec
from slime_adapter.megatron_hooks.moe_forward_patch import wrap_moe_layers
from slime_adapter.controller.switch_head import SwitchHead

def get_gpt_oss_with_gate_spec(args, config, vp_stage):
    spec = get_gpt_oss_spec(args, config, vp_stage)
    wrap_moe_layers(spec, num_layers=config.num_layers, hidden=config.hidden_size,
                    init_bias=args.gate_init_bias)
    return spec
```

`wrap_moe_layers` 在每个 MoE 层 build 时挂上 `SwitchHead` 实例（fp32 参数），并替换其 forward。

### 3.8 `slime_adapter/configs/gpt_oss_20b_8x5090.py`

参考 slime `tests/test_qwen3_30B_A3B.py` 的写法：

```python
NUM_GPUS = 8
PERF = (
    "--tensor-model-parallel-size 1 "
    "--pipeline-model-parallel-size 1 "
    "--expert-model-parallel-size 8 "
    "--expert-tensor-parallel-size 1 "
    "--context-parallel-size 1 "
    "--sequence-parallel "
    "--moe-token-dispatcher-type alltoall "          # 5090 无 nvlink, deepep 收益小
    "--recompute-granularity selective "
)

GATE = (
    "--gate-init-bias -2.0 "
    "--budget-fraction 0.7 "
    "--cache-window 16 "
    "--cache-cap 30 "
    "--budget-lambda 0.05 "                            # ① uniform per-switch cost 权重
    "--barrier-lambda 0.5 "                            # ② token-level hinge² barrier 权重 (硬上限)
    "--consistency-lambda 0.05 "                       # ③ chunk routing consistency 权重
    "--chunk-size 8 "                                  # chunk consistency 的窗口
    "--use-pressure-input 1 "                          # switch_head 用 pressure 当输入特征 (v3)
    "--top-k 2 "
)

ROLLOUT = (
    "--rollout-function-path slime.rollout.sglang_rollout.generate_rollout "
    "--rm-type custom "
    "--custom-rm-path slime_adapter.rollout.reward_kl:reward_func "
    "--rm-url http://localhost:30001 "                 # teacher SGLang
    "--use-routing-replay "
    "--rollout-num-gpus 8 --rollout-num-gpus-per-engine 4 "
    "--disable-cuda-graph "                            # 一期；二期再 graph-aware
)

GRPO = (
    "--advantage-estimator gspo "                       # slime 默认；我们 reward=0 → advantage 全 0
    "--use-tis "                                        # 与 slime 默认对齐
    "--num-update-epochs 1 "                            # 单 epoch 即可
)

MISC = (
    "--actor-num-nodes 1 "
    "--actor-num-gpus-per-node 8 "
    "--colocate "                                       # 8x5090 共宿主，单节点
    "--spec slime_adapter.spec:get_gpt_oss_with_gate_spec "
    "--budget-lambda 0.1 "
)
```

启动两条命令：① teacher SGLang server（`gpt-oss-20b` 原始权重）；② `python -m slime.train --config slime_adapter.configs.gpt_oss_20b_8x5090`。

---

## 4. Replay 协议（精简版）

| 量 | 是否 record | 谁负责 | 训练时怎么用 |
|---|---|---|---|
| `tokens` (B, T) | 是 | sglang_rollout | 直接喂 LLM forward |
| `teacher_log_probs` (B, T) | 是 | OPD reward_func | KL 中的 teacher 项 |
| `used_top2` (B, T, L, 2) | 是，slime `RoutingReplay` 自带 | SGLang select_experts hook | replay forward 时 `compute_topk` 强制返回这个 |
| `switch` (B, T, L) | 是，扩展 RoutingReplay | SGLang gate hook | 训练 forward 时**用作 STE 的 hard 路径**（forward 一致），但 σ 仍由 switch_head 当前权重产生 → 梯度可流 |
| `cache` 状态 | **不存**，从 `used_top2` 序列重建 | — | 训练 forward 时 layer_cache 离线扫一遍 |
| `n_new` (B, T, L) | **不存**，从 `used_top2` + `cache` 重建 | — | 用于 forward 中 σ × n_new 的 backward |

**关键澄清**：训练 forward 阶段，`switch` 的 forward 值由 record（保证 used_top2 一致）锁定，但 σ 是当前 switch_head 在新权重下重新算出来的——梯度 `∂L/∂σ` 通过 STE 传到 switch_head 参数。这是"hard forward + soft backward"的标准 STE 玩法。

**为什么仍然需要 RoutingReplay**：top2 是 router argmax，多 epoch 下 router 权重一变就漂；replay top2 保证训练 forward 走的是 rollout 那条 expert 路径，KL 才有意义。

**为什么 cache 不需要 record**：cache 是 `used_top2` 的滚动 union，确定性函数。record 了 used_top2 就等于 record 了 cache 的所有信息。

---

## 5. 训练步骤

逐 step 看一次完整 iteration：

1. **rollout**：student SGLang 生成 batch_size 条响应，每个 token 每层记录 `(switch, used_top2, n_new)`。teacher SGLang 单独跑一遍拿 `teacher_logp`。结果以 slime 的 `Sample` 对象回到 trainer。
2. **routing replay 装载**：slime 把 `used_top2` 写到 RoutingReplay 全局 buffer；switch / cache 状态由我们的 patch 在 `model_output` 上铺成 layer 维 list。
3. **Megatron training forward**：
   - 标准 prefill forward；`compute_topk` 走 RoutingReplay 返回 record 的 `used_top2`。
   - 我们的 MoE patch：算 σ_l = `sigmoid(switch_head(h))`；STE forward 等于 record 的 switch（保证 used_top2 一致），backward 给 σ 梯度。
   - cache 从 record 的 `used_top2_history` 重建（一次 O(T) 扫描，per layer）。
   - 算 `n_new`，累 `credits_used += switch_l * n_new_l`，`penalty_l = credits_used / total`。
   - 算 student logp 给 KL loss 用。
4. **Loss & backward**：
   ```
   L = mean(student_lp - teacher_lp)  +  λ · mean( Σ_l penalty(t,l) )
   ```
   纯 supervised，无 PG。backward 直接走完，optimizer.step()。
5. **权重同步**：slime 自带 `update_weights_from_distributed` 把 Megatron 的 σ_head + LoRA + router（如果训）推到 SGLang student worker。

teacher SGLang 永远不会被同步，weight 始终是原 gpt-oss。

---

## 4. Replay 严格性的几个边界

1. **cache 的 1-step lag**：第 t 个 token 的 cache 由 [t-16, t-1] 的 `used_top2` 决定。replay forward 沿 seq 维度做时，要在算 layer l 的 σ_l 前把 cache state 推到 t；这是 layer-level 串行常数（O(B·T·L)，全 Python 也飞快）。
2. **训练时 σ 与 rollout σ 不一致**：每个 epoch 都会让 σ 漂；但因为 forward 走 hard recorded switch，**前向计算结果 (used_top2, hidden, KL) 与 rollout 完全一致**，σ 的漂移只影响 backward 梯度 — 这是 STE 的预期行为，不是 bug。
3. **σ 漂得太多导致行为 mismatch**：如果训了几个 step 后，σ_new > 0.5 但 record 是 switch=0（或反之），STE forward 用 record（不切），而 σ_new 想切——这条 token 上 KL 实际还是 record 的 KL，σ 的 backward 梯度仍然通过 penalty 项给信号。**KL 那条路径在 STE forward 锁定 record 后没有"通过 σ 改变 KL 来改 σ"的反馈**。Penalty 仍然给 σ 反馈（让它对齐资源约束）。
4. 这意味着 **KL 主要靠 router 和 LoRA 学**（如果开 intra-option），σ 主要靠 penalty 学。两者通过共享 hidden state 间接耦合。
5. 想让 KL 真正反传到 σ：要么 (a) 不用 STE，每步重 forward 当前 σ 决策的 used_top2（off-policy 修正），(b) STE 的 forward 也用 σ_new 决策（重排实际过哪些 expert），代价是不能 reuse 推理时的 KV cache。**一期我建议接受这点限制**——当 σ 漂得太多时 record 自然过期，正常 multi-epoch on-policy 漂移不大；如果 KL 一直降不下来再考虑切到 (b)。

---

## 5. 5×5090 单节点资源预估

| 组件 | 显存 | 算力代价 |
|---|---|---|
| gpt-oss-20b bf16 主权重，EP=8 | 5 GB / 卡 | — |
| switch_head（24 层 × Linear(2880, 1)） | <1 MB | — |
| KV cache (rollout SGLang) | 8-12 GB | — |
| 训练 activation + selective recompute | 8-10 GB | — |
| Optim states (fp32 mom1+mom2 on switch_head + router LoRA + lm_head) | <1 GB | — |
| 余量 | 6-8 GB | 留给 batch 增长 |

bf16 全程，不量化、不 LoRA（除非你想把 router 当 trainable，那挂个 LoRA 也行）。**5090 没 nvlink**，所以：
- `--moe-token-dispatcher-type alltoall`（不开 deepep，节点内 PCIe gen5）
- TP=1，CP=1：避免 attention 内部走 PCIe all-gather
- EP=8 节点内：experts 切 8 份，每张 5090 持有 16 个 expert 权重 ≈ 1.5 GB
- 节点间（如果 N > 1 节点）：HSDP outer，IB 走 grad reduce-scatter

---

## 6. 实施 milestone

| 阶段 | 状态 | 内容 |
|---|---|---|
| **M0**  basic env  | ✅ done | clone slime + Megatron + sglang (`scripts/install_externals.sh`)；`uv sync`；smoke tests pass (`pytest tests/test_controller_core.py`) |
| **M1**  GRPO baseline | ✅ done | 在 8-GPU 上裸跑 slime GRPO/GSPO（不开 controller），mbridge / EP / SGLang 链路打通 |
| **M2**  controller forward | ✅ done | `moe_forward_patch.py` + `BatchedLayerCache`；rollout `request_state.py` 写满 records |
| **M3**  RoutingReplay | ✅ done | slime 的 `register_routing_replay_extensions` 与 Megatron 的 `compute_topk` 对齐；K=1 端到端 numerically equivalent |
| **M4**  loss + reward | ✅ done | `_apply_importance_weights_inplace` + `L_switch_pg` + barrier + chunk；cache cost 进 reward 路径已实现 |
| **M5**  p_mix rollout | ✅ done | `mix_generate.py`：student/teacher 双 SGLang 调用，per-token IS 权重；与 loss patch 互通 |
| **M6**  DeepSets context | ✅ done | `ExpertSetEncoder` + SwitchHead 接 `(cache_set, top_k_set)`；sequential T-loop 路径完成 |
| **M7**  end-to-end Qwen3-30B | 🔧 in progress | 8-GPU EP=8 真训：监控 switch_rate / cache_hit / n_new / KL / w_t 全套 metric；调 (α_q, α_c, β, λ_h) |
| **M8** | ⏳ next | CUDA graph 兼容、长 context、checkpoint resume、tensorboard 仪表盘 |

`slime_adapter/` 当前 ~2200 LoC（含测试），单元测试 35/35 pass（11 controller core
+ 3 integration mock + 7 p_mix + 7 batched cache + 7 deepsets switch head）。
真栈 4 个 e2e 测试（slime / Megatron / sglang / Qwen3）需要 GPU + 安装 external
依赖时跑。

## 7. 风险与待办

### 已落实（v4 当前实现）

| 项 | 状态 | 文件 |
|---|---|---|
| Cache cost 进 reward | ✅ | `rollout/reward_kl.py:post_process_rewards` |
| 联合 actor PG (`L_switch_pg`) | ✅ | `loss/penalty_loss.py:_wrapped_policy_loss` |
| p_mix rollout + IS 权重 | ✅ | `rollout/mix_generate.py:generate_one_p_mix` |
| IS 权重 rescale 进 advantage | ✅ | `loss/penalty_loss.py:_apply_importance_weights_inplace` |
| `BatchedLayerCache` (per-(b,t) 精确) | ✅ | `controller/cache_state.py:BatchedLayerCache` |
| DeepSets context 进 SwitchHead | ✅ | `controller/expert_set_encoder.py` + `switch_head.py` |
| Teacher logprob 拉取（KL anchor 数据源）| ✅ | `rollout/mix_generate.py:_fetch_teacher_logprobs_full` 一次性 post-rollout |
| `chunk_consistency` 函数接口 | ✅ stub | `loss/chunk_consistency.py:compute_chunk_consistency` — 当前返回 0，adapter 接通 `router_logits` 后翻 `enabled=True` |
| 冷启动 cache 不收 cost | ✅ | `reward_kl.trajectory_cache_cost(skip_first=...)`；config knob `cache_cost_cold_start_skip` |
| KL 单一来源（不双计）| ✅ | 仅 slime `--use-kl-loss --ref-load <teacher>`；slime 自检 assert `kl_coef` 与 `kl_loss_coef` 互斥 |

### 仍需 M7 实跑验证 / 后续待办

1. **slime batch 的 metadata 透传**：`_apply_importance_weights_inplace` 既兼容 `batch['importance_weights']` 也兼容 `batch['samples'][i].metadata['importance_weights']` 两条路。slime 实际把 Sample 序列化进 RolloutDataRef 时会不会保留 `metadata` 字段需要 M7 短训校验。如果丢了，加一个 `compute_advantages_and_returns` 之后的 hook 把 sample.metadata 同步到 batch dict。
2. **STE × p_mix × PG 三路梯度的方向一致性**：SwitchHead 的梯度同时来自 PG 项 (`A_t · log π(switch)`)、STE 经 KL 的 backward、以及 barrier。三者方向应趋同（"该切就切，预算紧别切"），但 advantage 噪声 + STE 不连续可能导致 σ 抖动。建议短训观察 σ trajectory；必要时给 `switch_pg_lambda` 做 warm-up（前 N 步 = 0，让 KL+barrier 先把 σ 钳住）。
3. **SGLang cuda graph 兼容**：v1 关 graph，吞吐损失约 20%。后续上 fixed-shape mask + pad-to-cap 让 graph 可重入。
4. **5090 / EP=8 下 alltoall 才是真瓶颈？**：cache-cost 公式假设瓶颈是"重新加载新 expert 的权重"。如果你的部署里 alltoall token traffic 才是主要开销，cost 公式应改成 `traffic(used_top2)` 而不是 `n_new`。M7 之前需对齐。
5. **Reward hacking 监控**：M7 训练曲线上要同时看 `cache_cost_raw`、`response_length`、`task_correctness`、`KL_to_teacher`、`is_weight_mean`。健康表现：cache_cost 缓降 + KL 缓降 + length 稳定 + correctness 升。塌降表现：cache_cost 急降 + length 急降 + correctness 不动 → token-空间 hacking。对策：升 `teacher_mix_alpha` 或加 length-floor 正则。
6. **`L_chunk_consistency` 现阶段是 0**：占位接口已就位 (`compute_chunk_consistency(..., enabled=False)`)；启用需要在 adapter 的 forward 里把 `router_logits` 传给该 hook 并写到 `state.chunk_consistency_loss`。Qwen3 adapter 待补。`λ_chunk` 在 config 里保留但当前实际无效。
7. **`data_source` 协议**：`mix_generate.generate_rollout` 假设 slime 给的 `data_source` 是 dict-yielding 可迭代；slime 实际用 `RolloutDataSourceWithBuffer`，里面有 buffer / partial 等机制。`_iter_prompts` 是 best-effort 兼容层，M7 时再核一次。
8. **冷启动**：现已支持 `cache_cost_cold_start_skip`，前 N 个 response token 不计 cache 成本（Yaml 默认 16 = window 大小）。如果实测 N=16 仍偏严，下一步还可以做 `i/window` 的线性折扣。
9. **batched cache 的 cap 触发性能**：`BatchedLayerCache.push` 在超 cap 时调 `(count > 0).sum().item()` 会 CPU sync；正常 window·k ≤ cap 时不进入这条路径，但要在 M7 真实序列上 profile 一次。

'''


## 8. 一句话

GRPO 的 actor 同时驱动 token 输出和 per-layer 的 SwitchHead 决策，cache 状态 ω_t 是 MDP 的一部分；reward 显式包含 cache cost，让 PG 学会"现在花一点票把 d 装进 cache，将来 t+5 复用 d 时省回来"。SwitchHead 通过 DeepSets 看到当前 cache 集合 + router 想要的 top-K，做出局部最优的切换决策。Rollout 用 MiniLLM p_mix 防 token-空间 reward hacking，IS 权重在 loss 端 rescale。整套实现保留 slime 的 GRPO 主线、Megatron 的 EP 训练、SGLang 的 batched 推理，约 2200 LoC + 35 个单测全过。
