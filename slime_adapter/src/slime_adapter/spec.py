"""Megatron ``--spec`` entry — wires our controller into the model build path.

Used as::

    --spec slime_adapter.spec:get_spec_with_controller

The function is invoked by Megatron's spec resolution after CLI args are
parsed but before the model is built. It returns the same kind of layer-spec
object that Megatron's stock ``--spec`` returns, possibly with extra hooks
attached.

For Qwen3-MoE, we re-use slime's existing spec entry point and additionally
register our adapter so subsequent training-loop calls can find it via
``slime_adapter.modeling.get_adapter("qwen3_moe")``.

The actual installation of ``SwitchHead`` / cache / budget into the built
model happens AFTER model construction (since we need the live module
references), in
``slime_adapter.megatron_hooks.install_controller_into_layers(model, args)``.
The user should call that from their training entrypoint right after model
build.
"""

from __future__ import annotations

from typing import Any


def get_spec_with_controller(args, config, vp_stage=None):
    """Return a layer spec with our controller-aware MoE wiring.

    For Qwen3-MoE, we delegate to slime's stock spec (slime doesn't ship a
    Qwen3 spec file by default — depending on version, this could be
    ``slime_plugins.models.qwen3_moe`` or live in Megatron-bridge). For now
    we expect the user has wired their own ``--spec`` and call this function
    on top to get the controller in.

    Returns: the spec object (same type as the underlying delegate).
    """
    # The exact import depends on slime/Megatron-bridge versions. Try a few
    # candidates and fall through.
    base_spec = _resolve_base_spec(args, config, vp_stage)

    # Register a marker on the spec so the trainer can sanity-check that
    # the slime_adapter forward patch was installed. We don't actually
    # mutate the spec here — install happens post-build via
    # ``install_controller_into_layers(model, adapter, args)``.
    if hasattr(base_spec, "__dict__"):
        base_spec._slime_adapter_controller_intended = True

    return base_spec


def _resolve_base_spec(args, config, vp_stage):
    """Find the underlying Megatron spec callable for this model."""
    # First try: a slime_plugins spec for the architecture.
    arch = getattr(args, "moe_arch", "qwen3_moe")
    candidates = [
        f"slime_plugins.models.{arch}",      # e.g. slime_plugins.models.qwen3_moe
        f"slime_plugins.models.{arch}_moe",  # fallback
    ]
    for mod_path in candidates:
        try:
            mod = __import__(mod_path, fromlist=["get_spec"])
        except ImportError:
            continue
        for fn_name in ("get_spec", f"get_{arch}_spec"):
            fn = getattr(mod, fn_name, None)
            if fn is not None:
                return fn(args)

    # Last resort: assume the user passes a fully-qualified spec via another flag.
    raise RuntimeError(
        f"Could not find a base layer spec for arch={arch!r}. "
        f"Pass an explicit ``--base-spec`` or implement "
        f"``slime_plugins.models.{arch}.get_spec``."
    )
