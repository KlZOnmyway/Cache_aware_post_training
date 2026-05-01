"""Trainer → SGLang weight sync for SwitchHead modules.

SGLang's MoE select hook needs the per-layer ``SwitchHead`` to be present in
the SGLang worker's process. Since the SwitchHead is trained on the Megatron
side, every time slime broadcasts updated weights to SGLang we also need to
push the SwitchHead parameters across.

This module provides:

  - ``serialize_switch_heads(model, adapter)`` — trainer-side: pull the
    SwitchHead state_dicts off ``model``'s MoE layers and pack them into a
    cpu-resident dict { layer_idx → (state_dict, hidden_size, use_pressure_input) }.
  - ``deserialize_and_register(payload)`` — sglang-side: rebuild the
    ``SwitchHead`` modules from the payload and register them via
    ``slime_adapter.sglang_patches.register_switch_head``.
  - ``send_to_sglang_url(url, payload)`` / ``recv_from_trainer(...)`` —
    minimal HTTP transport for environments where Ray / NCCL aren't a fit.

Slime's standard weight-sync path (``actor_model.update_weights()``) already
broadcasts the trainer's full state_dict via NCCL. The cleanest hook is to
piggyback on that: after ``update_weights()`` returns, the trainer also calls
``broadcast_switch_heads(...)``. See ``examples/wiring_into_slime.py`` for
the recommended wiring.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Tuple

import torch

from slime_adapter.controller.switch_head import SwitchHead
from slime_adapter.modeling._base import MoEModelAdapter

# We import the bank lazily inside the deserialize path so this module stays
# usable in trainer-only environments where the SGLang patch isn't loaded.


# ---------------------------------------------------------------------------
# Trainer side: pull SwitchHeads off the model
# ---------------------------------------------------------------------------

def serialize_switch_heads(
    model,
    adapter: MoEModelAdapter,
) -> Dict[int, Dict[str, object]]:
    """Walk MoE layers, pull each SwitchHead state_dict + metadata.

    Returns a CPU-tensor dict keyed by layer index. The result is suitable for
    pickling / NCCL broadcast.

    Each entry contains:
        {
            "state_dict": OrderedDict[str, Tensor],
            "hidden_size": int,
            "use_pressure_input": bool,
            "init_bias": float,
        }
    """
    out: Dict[int, Dict[str, object]] = {}
    for handle in adapter.iter_moe_layers(model):
        head: SwitchHead | None = getattr(handle.module, "switch_head", None)
        if head is None:
            raise RuntimeError(
                f"Layer {handle.layer_idx} has no .switch_head attribute. Was "
                f"install_controller_into_layers(model, adapter, args) called?"
            )
        # Move state_dict to CPU; SGLang will move it back to its own device.
        state = {k: v.detach().to("cpu") for k, v in head_state_dict(head).items()}
        out[int(handle.layer_idx)] = {
            "state_dict": state,
            "hidden_size": int(head.hidden_size),
            "use_pressure_input": bool(head.use_pressure_input),
            "init_bias": float(head.init_bias if hasattr(head, "init_bias") else -2.0),
        }
    return out


def head_state_dict(head: SwitchHead) -> dict:
    """Return ``head.state_dict()`` keyed without leading prefixes."""
    return {k: v for k, v in head.state_dict().items()}


# ---------------------------------------------------------------------------
# SGLang side: rebuild + register
# ---------------------------------------------------------------------------

def deserialize_and_register(
    payload: Dict[int, Dict[str, Any]],
    *,
    target_device: str | None = None,
    target_dtype: "torch.dtype | None" = None,
) -> int:
    """Rebuild SwitchHead modules from payload and register them.

    Args:
        payload: dict produced by ``serialize_switch_heads``.
        target_device: where to put the rebuilt modules; defaults to ``cuda``
            if available, else ``cpu``.
        target_dtype: optional dtype to cast parameters to. Default keeps fp32
            which is what the trainer ships.

    Returns:
        Number of heads registered.
    """
    # Lazy import so this works even if sglang isn't installed in the trainer env.
    from slime_adapter.sglang_patches.moe_select_patch import register_switch_head

    if target_device is None:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

    n = 0
    for layer_idx, entry in payload.items():
        sd = entry["state_dict"]
        head = SwitchHead(
            hidden_size=int(entry["hidden_size"]),
            init_bias=float(entry.get("init_bias", -2.0)),
            use_pressure_input=bool(entry.get("use_pressure_input", True)),
            zero_init_weight=False,  # weights will be overwritten by load_state_dict
        )
        head.load_state_dict(sd, strict=True)
        if target_dtype is not None:
            head = head.to(dtype=target_dtype)
        head = head.to(device=target_device)
        head.eval()  # rollout side: no autograd
        register_switch_head(int(layer_idx), head)
        n += 1
    return n


# --------------------------------------------------------------------------
# Optional HTTP transport (simplest path when NCCL group is not available)
# --------------------------------------------------------------------------

def serialize_to_bytes(payload: Dict[int, Dict[str, Any]]) -> bytes:
    """Pickle a payload dict to bytes (uses torch.save for tensor support)."""
    buf = _BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def deserialize_from_bytes(blob: bytes) -> Dict[int, Dict[str, Any]]:
    return torch.load(_BytesIO(blob), map_location="cpu", weights_only=False)


def _BytesIO(*args, **kwargs):  # tiny indirection so we don't add an import at top
    import io
    return io.BytesIO(*args, **kwargs)


# --------------------------------------------------------------------------
# Convenience: full-loop sync (trainer side)
# --------------------------------------------------------------------------

def push_to_sglang_servers(model, adapter, sglang_urls: List[str]) -> None:
    """Trainer-side helper: serialize + push to a list of SGLang servers via HTTP.

    Each SGLang server must expose a ``/load_switch_heads`` endpoint that
    receives the serialized blob and calls ``deserialize_and_register``.

    For an NCCL-based path, replace this with a direct broadcast — the
    serialized state_dict is small (~< 1 MB total even for 60 layers).
    """
    import requests  # imported lazily; not in core deps

    payload = serialize_switch_heads(model, adapter)
    blob = _to_bytes(payload)
    for url in sglang_urls:
        resp = requests.post(f"{url.rstrip('/')}/load_switch_heads", data=blob, timeout=30)
        resp.raise_for_status()


def _to_bytes(payload: Dict[int, Dict[str, Any]]) -> bytes:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

class _BytesIO_StubForOlderEnvs:
    pass
