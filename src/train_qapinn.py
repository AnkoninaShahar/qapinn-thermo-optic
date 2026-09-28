from __future__ import annotations

try:
    from .train import train_one
except ImportError:  # pragma: no cover
    from train import train_one


__all__ = ["train_one"]
