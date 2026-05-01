# 适配 slime 方案文档（v3: STE distillation + uniform per-switch cost + budget barrier + chunk-routing consistency）

> **v2 改动说明（vs v1）**
> v1 把现仓库的 Option-Critic / PL selection / V+Q heads / GAE 整套搬过去；v2 根据后续算法讨论，方案简化为：
> - 唯一可学动作 = 每层一个 sigmoid `switch_head`；用 STE 把 forward 的二值化和 backward 的连续梯度桥接，**不做 Bernoulli 采样、不跑 REINFORCE**。
> - top2 的"new option"= router argmax，**不再有 selection 头 / Plackett-Luce**。
> - 状态 = per-layer rolling cache（最近 16 token 的 used_top2 union，cap 30 expert）+ per-token credit budget（`0.7 × num_moe_layers`，**每 token 重置**）。
> - Loss = `KL(student||teacher) + λ · Σ_l penalty(t,l)`，纯 supervised distillation。**不再有 Option-Critic、V/Q heads、advantage、GAE、selection PG**。
> 整个 trainer 复杂度从 RL 跌回到"distillation + auxiliary penalty"，对应工程量大幅下降。
>
> **v3 修订（在 v2 基础上）**：把 budget penalty 从 `(1 + α · pressure)` 形式改回 **uniform `switch · n_new`**，去掉 layer-index 偏见；`pressure` 改作 `switch_head` 的 forward-only 输入特征；引入可学 `per_layer_bias` 让模型自适应层级先验；新增 token-level **hinge² hard barrier** 防止超出 0.7L 总预算；保留 **chunk routing consistency** loss 让 router 时序平滑。Layer 间分配完全交给 KL gradient 通过 hidden chain 的回传 + budget barrier 触发的均摊压力来决定，模型可以学到 U-shaped / bell-shaped 等任意层级需求分布。

---

## 0. TL;DR

- 把 slime 当三方库装（`pip install -e ../slime`），所有适配代码集中在新增的 `slime_adapter/` 包里。**不 fork slime，不改 slime 源码**。
- Megatron-LM 和 SGLang 各打一个本地 patch（slime 已经为 Megatron 提供 patch 模板，参考 `docker/patch/latest/megatron.patch`）。
- 基座保留 gpt-oss-20b：slime 已自带 `slime_plugins/mbridge/gpt_oss.py` 和 `slime_plugins/models/gpt_oss.py`。
- 算法范式：**带预算约束的 conditional-compute distillation**，不是 RL。STE 把 sigmoid gate 的 hard forward / soft backward 缝合起来，long-horizon credit assignment 不需要——因为 credits 每个 token 都重置，跨 token 的耦合只通过离散的 cache 状态发生。
- 5090 部署用 EP=8 + DP，bf16，不开 LoRA / 不量化（gpt-oss-20b 在 EP=8 后每卡 ≈ 5GB 主权重，余量充足）。

---

## 1. 算法形式（参考实现，便于对照）

每个 token t、每层 l 在 forward 时：

```text
hidden_t,l                                     # 来自上一子层
σ_l       = sigmoid(switch_head(hidden_t,l, pressure_prev))   # ∈ [0,1]
                                                # switch_head 输入: (hidden, pressure_at_entry)
                                                # 输出 logit + per_layer_bias_l，sigmoid 得 σ
switch_l  = STE(σ_l)                            # forward: I[σ>0.5]; backward: ∂switch/∂σ ≈ 1
                                                # 见 slime_adapter/controller/ste.py

new_top2  = topk(router_logits_t,l, k=2)        # 路由的"如果切换"候选
used_top2 = switch_l ? new_top2 : current_top2  # current_top2 来自上一 token 的 used
n_new     = | new_top2 \ cache_t,l |            # 0,1,2 — cache 由过去 16 token 的 used_top2 union

# 预算账（per token，从 layer 0 累加，token 边界重置）
credits_used_so_far += switch_l * n_new          # 仅 switch_l 是带梯度的标量，n_new 是离散常量

# pressure 仅作为 switch_head 的输入特征 (forward-only, detached)
pressure_l = (credits_used_so_far / total_credits).detach()
             # 不进 cost 公式 → 不会引入"早层切便宜 / 晚层切贵"的人为偏见
             # total_credits = 0.7 × num_moe_layers, per-token reset
```

token 末尾：每层把 `used_top2_l` 推入 `cache_deque_l`，`>16` evict 最早的；重新 union 得 `cache_(t+1, l)`，cap 在 30 expert（极少触发）。

每 token 的训练目标：

```
L_token_t = - KL(student_t || teacher_t)                                                 # output distillation
          + λ_b · Σ_l  switch(t,l) · n_new(t,l)                                            # ① uniform per-switch cost (no layer bias)
          + λ_h · max(0, Σ_l switch(t,l)·n_new(t,l) − total_credits)²                     # ② token-level hard barrier
          + λ_c · L_chunk_consistency(student_router_logits, chunk_size=K)                # ③ chunk-wise routing 平滑

# pressure(t, l) = credits_used_before_l(t, l) / total_credits 不再进 cost，
# 改成 switch_head 的输入特征（forward-only，让 σ 决策时显式看到 budget 状态）。
# 这样 cost 是 layer-uniform 的，避免人为给早层 / 晚层加偏见——
# layer 间"先紧后松"或"先松后紧"完全由 KL gradient 自适应决定。
# detach 保证：每层 σ 收到的 gradient 仅来自本层自己的 switch·n_new 项，
# 前层 switch 通过改变 pressure 的影响被切断 → 不会双重计费
```

Reward / advantage / GAE / Q heads / V heads / Plackett-Luce / Bernoulli 采样**全部不存在**。loss 直接 backward。

**Per-layer 梯度结构**（关键）：pressure 在使用时被 detach，每个 σ(t, l) 收到的 budget gradient 完全 local：

```
∂L_total / ∂σ(t, l)  ≈  ∂L_KL / ∂σ(t, l)        # 穿过 hidden chain，含"后续层影响"
                       + λ_b · n_new(t, l)         # uniform per-switch cost
                       + λ_h · 2·overflow·n_new    # 超预算时 barrier 启动（全员均摊）
                       + λ_c · ∂L_chunk / ∂σ        # router 时序平滑
```

**关键设计原则**：cost 故意不带 layer-index 偏见（不再有 `(1 + α · pressure)` 那种"晚切贵"的强假设），因为层间需求分布是 **task / model 决定**，先验上既不是单调递增也不是递减——浅层、中层、深层都可能是关键瓶颈。layer 间"哪层 reserve / 哪层多花"的分配让 **KL 梯度通过 hidden chain 自然反传决定**：早层的 σ 会同时收到来自所有后续层 KL 的合并梯度，所以早层"知道"后面层重要不重要——不需要显式 lookahead。

`pressure` 改作 `switch_head` 的输入特征（forward-only），让 σ 决策时显式看到当前预算状态：

```python
class SwitchHead:
    def forward(hidden, pressure_scalar):     # pressure 只入 input, 不入 loss
        x = concat(hidden, pressure_scalar.unsqueeze(-1))
        return Linear(x).squeeze(-1) + per_layer_bias  # per_layer_bias 可训, 学层级先验
```

每层一个可学 `per_layer_bias`：让模型自适应学到"我这层一般该切 / 不该切"的先验，把层间重要性差异挪到这个 bias 上而不是塞到 cost 公式里。Token-level barrier 是唯一的硬约束执行器；soft cost 仅起"别滥切"的弱正则作用。

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

```python
import asyncio, aiohttp, torch
from slime.utils.types import Sample

async def reward_func(args, sample: Sample, **kwargs):
    """对接 teacher SGLang server，拿 token-level logprobs。"""
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {"max_new_tokens": 0, "logprobs": True},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(args.rm_url, json=payload) as resp:
            return await resp.json()

def post_process_rewards(args, samples, **kwargs):
    """组装 sample.teacher_log_probs 给 loss 端用，标量 reward 直接置 0。"""
    for sample, raw in zip(samples, raw_rewards):
        teacher_lp = torch.tensor(
            [t[0] for t in raw["meta_info"]["input_token_logprobs"][1:]]
        )[-sample.response_length:]
        sample.teacher_log_probs = teacher_lp
    return [0.0] * len(samples), [0.0] * len(samples)
```

KL 项不在 reward 里算（reward = 0），统一在 loss 端从 student / teacher logprob 直接构造。这和 slime 自带 `on_policy_distillation.py` 同构。

### 3.6 `slime_adapter/loss/penalty_loss.py`

import-time 替换 slime 的 `policy_loss_function`：

```python
import torch
import slime.backends.megatron_utils.loss as _slime_loss

_orig_policy_loss = _slime_loss.policy_loss_function

def patched_policy_loss(rollout_data, model_output, *args, **kwargs):
    base = _orig_policy_loss(rollout_data, model_output, *args, **kwargs)

    # 1. KL 项（OPD 风格）
    student_lp = model_output.logp_per_token  # [B, T]
    teacher_lp = rollout_data.teacher_log_probs  # [B, T]
    kl_per_tok = (student_lp - teacher_lp).clamp(min=-20, max=20)
    L_kl       = kl_per_tok.mean()  # 注意这是 sampled KL，可正可负

    # 2. Penalty 项: per-layer per-token credits 的累加
    # MoE patch 在 forward 时把每层 penalty 推到 model.layer_penalties: list[Tensor[B,T]]
    # 每层 forward 时算好 layer_local_costs[l] = switch(t,l) · n_new(t,l)（v3: layer-uniform，无 pressure 系数）
    # pressure 改作 switch_head 的 input feature（forward-only），不再进 cost
    layer_costs = getattr(model_output, "layer_local_costs", None)   # list[Tensor[B,T]]
    if layer_costs is not None:
        L_pen = torch.stack(layer_costs, dim=0).sum(dim=0).mean()
    else:
        L_pen = torch.zeros((), device=L_kl.device)

    # 3. Token-level barrier (hinge²): 防止"分摊到很多层每层小切"绕过总预算
    overflow = getattr(model_output, "budget_overflow_per_token", None)  # max(0, used - total)
    L_barrier = (overflow ** 2).mean() if overflow is not None else torch.zeros((), device=L_kl.device)

    # 4. Chunk routing consistency (router 时序平滑 → cache 自然命中)
    L_chunk = getattr(model_output, "chunk_consistency_loss", torch.zeros((), device=L_kl.device))

    L_total = L_kl + args.budget_lambda * L_pen + args.barrier_lambda * L_barrier + args.consistency_lambda * L_chunk
    base.loss = L_total
    base.metrics.update({
        "loss/kl": L_kl.item(),
        "loss/layer_local_cost": L_pen.item(),
        "loss/token_barrier": L_barrier.item() if isinstance(L_barrier, torch.Tensor) else 0.0,
        "loss/chunk_consist": L_chunk.item() if isinstance(L_chunk, torch.Tensor) else 0.0,
    })
    return base

_slime_loss.policy_loss_function = patched_policy_loss
```

替换在 `train.py` 顶部 `import slime_adapter.loss.penalty_loss` 一次即生效。

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

| 阶段 | 周数 | 交付 |
|---|---|---|
| **M0** | 0.5 | clone slime + Megatron + sglang，pip install -e；跑通 `tests/test_qwen3_30B_A3B.py` 当 smoke test |
| **M1** | 0.5 | gpt-oss-20b 在 8×5090 上 vanilla GRPO 跑通（验证 mbridge / EP / SGLang 路径都好使，**不开 controller**） |
| **M2** | 1.0 | `--use-routing-replay` 跑通；teacher SGLang server 起来；`reward_kl` 拿到合理 KL 数值 |
| **M3** | 1.5 | `slime_adapter/sglang_patches/moe_select.py` 注入 SGLang，rollout 输出带 `(switch, used_top2, n_new)`；switch 全置 0/1 做 sanity（recover 原 routing） |
| **M4** | 1.5 | `slime_adapter/megatron_hooks/moe_forward_patch.py` 完成；training forward 拼起来；σ 与 record switch 对齐验证 |
| **M5a** | 0.5 | `penalty_loss.py` 注入：`L = KL + λ_b·uniform_cost + λ_h·barrier + λ_c·consistency`；端到端跑通；监控 switch_rate / cache_hit / KL 三件套 |
| **M5b** | 0.5 | 调 `λ_b / λ_h / λ_c` 与 `init_bias`；ablation：use_pressure_input=False、关掉 layer_bias、关掉 chunk consistency 各跑一组对照 |
| **M6** | 1.0 | CUDA graph 兼容、长 context、checkpoint resume、tensorboard metrics 等收尾 |

合计 **6 周**，与 v1 相比少 1-2 周（因为没了 Option-Critic / V/Q / PL 那一坨）。

---

## 7. 风险与待办

1. **STE bias**：KL 通过 hidden chain 反传时，对 STE 桥接处会有偏差，本质是 `1{σ>0.5}` 的不可微近似。一般实践里可接受；如果实测 controller 不收敛，备用方案：
   - (a) 训练时按 σ 直接 forward "soft mix"（同时算 cache top2 与 new top2，按 σ 加权）；2× compute 但梯度无 bias。仅作 ablation。
   - (b) 加入轻量 Bernoulli + REINFORCE 给 σ 一条带 GAE 的跨 token 信号通路（混合 STE 短链 + PG 长链）。
2. **SGLang cudagraph**：一期关 graph，吞吐损失约 20%。后续做 fixed-shape mask 让 graph friendly。
3. **EP=8 + cache locality**：每张卡只持有 1/8 的 expert 权重；当 used_top2 跨 rank 时 dispatch 仍走 all-to-all。"cache 命中"这件事**只省载入新权重的开销，不省 token routing 的通信**。如果你的"load 一个 expert"成本模型其实是后者，penalty 公式就要重定义。**这个语义假设需要在 M0 之前先和算法同学对齐**。
4. **First 16 tokens cold start**：cache 空 → n_new 必然为 2 每层 → credits 一次烧完。建议：
   - prompt 段（query 段）不计 penalty（mask 掉），只对 response 段计算；
   - 或者第 1-16 个 response token penalty 折扣 (16-i)/16。
5. **Penalty scale (v3)**：三个 λ 互相耦合，建议起手：
   - `λ_b = 0.05`（uniform per-switch cost）：每层 switch · n_new ∈ [0, 2]，24 层求和量级 ~0-48，per-token mean ~10-20，乘 0.05 = 0.5-1.0
   - `λ_h = 1.0`（barrier 平方，仅在超预算时激活）：overflow ∈ [0, ~10]，平方后 ~0-100，乘 1.0 给硬约束足够压力
   - `λ_c = 0.05`（chunk consistency, KL on softmax）：~0.1-0.5 数量级
   - 总 `L_aux ≈ 1-3`，与 `L_KL ≈ 5` 同量级。先 fix 跑一组，再 grid。
6. **Per-token credit reset 是显式还是隐式**：CreditsTracker.reset_per_token，在 MoE patch 检测到 token_idx 变化时调一次。注意这要求每层 patch 知道当前 token 索引——可以从 `attention_mask` 或 `position_ids` 推。
7. **gpt-oss `top_k=4` 的默认 vs 我们要的 `top_k=2`**：mbridge 配置里要显式设 `moe_router_topk=2`，否则 router 会输出 4 个 expert。
8. **Cache-policy 耦合 / chicken-and-egg（重要风险）**：cache 在 Mode A（`cache.push(used_top2)`）下随 switch policy 漂移，导致 loss landscape 非平稳。可能后果：
   - **Dead-on-init**：σ 初始偏低 → 几乎不切 → cache 永远停在初始 top2 → 模型在劣质 cache 上学到 stuck 的 σ。
   - **Cache thrashing**：σ 偏高 → 频繁切换 → cache 永远刷新 → 复用率 0，与不开 controller 等价。
   - **局部最优锁死**：(σ, cache) 进入互相强化的稳定点，跳不出来。
   缓解措施（按优先级）：
   - **必做** (a) `λ-warmup`：前 N 步 λ=0，让 cache 在 KL 信号下自然成型；再线性升到目标 λ。
   - **必做** (b) `init_bias` 适中（建议 0.0 ~ -1.0，σ ∈ [0.27, 0.5]），避免 σ 一开始就极端偏。
   - **必做** (c) 跑 **Mode B 对照**（`cache.push(router_top2)` 而非 `used_top2`），cache 演化与 switch 政策解耦但仍随 student router 漂移。**v3 推荐再加一个 Mode C: Teacher-anchored cache** （`cache.push(teacher_router_top2)`）。teacher router 永不更新 → cache trajectory 在整个训练里固定 → loss landscape 静态，chicken-and-egg 彻底消失。student router / switch_head 都正常训，frozen 的只是 teacher 自带的 router。penalty 改为 `|student_top2 \ teacher_cache|`，自然把 student routing 朝 teacher 偏好对齐，等价于 routing-level 的对齐正则。建议作为 Phase-1 默认配置。Mode C → Mode A fine-tune 是 Phase-2 的过渡路径（关闭 teacher cache，让 student 自洽起来对齐部署语义）。

   - 关于 student router 是否 frozen：**只有 teacher router 是 frozen 的**（teacher 本来就是 frozen 整个模型）；student router 全程可训，KL + cache penalty 一起驱动它收敛到"和 teacher 等价但只用 top-2"的解。可选加 router KL anchor `η · KL(student_router_logits || teacher_router_logits.detach())` 把 student router 拉在 teacher 邻域，更稳。

   - 实现：teacher SGLang server 在 OPD 接口多返回 `router_top2_per_layer: tensor[T, L, 2]`，rollout 一次性记到 `Sample.metadata['teacher_top2']`；Megatron 训练 forward 时按这份 record 重建 cache，与 routing replay 共用同一份序列。`slime_adapter/megatron_hooks/moe_forward_patch.py` 加配置 `--cache-source {teacher,student_router,used}`。
   - Train-inference gap：部署时无 teacher，可以 (i) 退化到 Mode A 用 student 自己的 used_top2（信任训练已经把 student routing 对齐到 teacher），或 (ii) 部署时仍保留一个 frozen teacher router 副本（仅一层 `Linear[H, E]`，加载几十 MB 参数）专门生成 cache，开销可忽略。
   - **可选** (d) **Lookahead consistency loss**：`L_smooth = mean‖student_router_logits[t] - student_router_logits[t-1]‖²` 或类似 token-邻接 KL，鼓励 routing 时序平滑，配合 cache locality。
   - **可选** (e) **Chunk-wise switch**：让 `switch_head` 每 K=4 个 token 决定一次，期间复用同一 switch 决策；强制时间相干，cache 抖动率降到 1/K。
   - **可选** (f) **Teacher prefilled cache**（cold-start）：rollout 的前 16 token 用 teacher top-2 预填 cache，避免冷启动期 n_new 全 2 把预算烧光。Mode C 下天然成立（cache 一直来自 teacher）。
   - **必做监控**：训练 metric 加 `cache_diversity / 30`、`switch_rate_per_layer`、`student_top2_overlap_with_teacher_cache`、`n_new_distribution`、`KL_along_sequence`，任一跑歪立刻识别 chicken-and-egg / cache 死锁。

---

## 8. 一句话

把整套 RL/Option-Critic/PL 那一坨退化成：**一个 sigmoid switch_head（输入 hidden + pressure 特征）用 STE 训，loss = `KL(student||teacher) + λ_b·Σ switch·n_new + λ_h·hinge²(overflow) + λ_c·chunk_consistency`**。每层一个可学 layer_bias 自适应学层级先验；KL gradient 通过 hidden chain 自然完成跨层 credit assignment；token-level barrier 兜底硬约束；per-token credit reset，跨 token 耦合只通过离散 cache 状态。其余全部交给 slime + Megatron + SGLang 既有路径。整个 `slime_adapter/` 估计 < 1500 行 Python，外加 Megatron / SGLang 各一个百行级 patch。
