"""Tensorboard logger — minimal wrapper around ``torch.utils.tensorboard.SummaryWriter``.

Replaces the legacy wandb-based logging path. Use it like::

    from slime_adapter.utils.tb_logger import get_logger

    logger = get_logger(log_dir="runs/exp1")
    logger.log_scalar("loss/total", 0.5, step=42)
    logger.log_scalars("loss", {"kl": 0.3, "budget": 0.1, "barrier": 0.0}, step=42)
    logger.flush()

Multi-process: only rank 0 actually writes; other ranks silently no-op.
"""

from __future__ import annotations

import os
from typing import Any, Optional

_LOGGER: Optional["TBLogger"] = None


class TBLogger:
    """Thin SummaryWriter wrapper. Idempotent get_logger() returns the singleton."""

    def __init__(self, log_dir: str, *, rank: int = 0, flush_secs: int = 30):
        self.log_dir = log_dir
        self.rank = int(rank)
        self.enabled = self.rank == 0
        self.writer = None
        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as e:
                raise RuntimeError(
                    "tensorboard not installed; pip/uv install tensorboard"
                ) from e
            self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        if not self.enabled:
            return
        try:
            self.writer.add_scalar(name, float(value), int(step))
        except Exception:
            pass

    def log_scalars(self, prefix: str, values: dict[str, float], step: int) -> None:
        if not self.enabled:
            return
        for k, v in values.items():
            self.writer.add_scalar(f"{prefix}/{k}", float(v), int(step))

    def log_text(self, name: str, text: str, step: int = 0) -> None:
        if not self.enabled:
            return
        self.writer.add_text(name, text, int(step))

    def log_histogram(self, name: str, values, step: int) -> None:
        if not self.enabled:
            return
        try:
            import torch
            t = values if hasattr(values, "detach") else torch.tensor(values)
            self.writer.add_histogram(name, t.detach().cpu(), int(step))
        except Exception:
            pass

    def flush(self) -> None:
        if self.enabled and self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.enabled and self.writer is not None:
            self.writer.close()


def get_logger(log_dir: str | None = None, *, rank: int = 0) -> TBLogger:
    """Return the process-singleton logger; first call sets ``log_dir``."""
    global _LOGGER
    if _LOGGER is None:
        if log_dir is None:
            log_dir = os.environ.get("SLIME_ADAPTER_TB_DIR", "runs/default")
        _LOGGER = TBLogger(log_dir=log_dir, rank=rank)
    return _LOGGER


def reset_logger() -> None:
    """Drop the singleton (useful in tests)."""
    global _LOGGER
    if _LOGGER is not None:
        _LOGGER.close()
        _LOGGER = None


__all__ = ["TBLogger", "get_logger", "reset_logger"]
