"""splatgif: one photo in, a 3D "wigglegram" GIF out."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.1.0"

if TYPE_CHECKING:
    from .gif import save_gif  # noqa: F401
    from .motion import MOTIONS  # noqa: F401
    from .pipeline import GifInfo, make_gif  # noqa: F401
    from .splat import Splat, predict  # noqa: F401

# Resolved on first use so that `splatgif --help` does not pay for importing torch.
_LAZY = {
    "MOTIONS": "motion",
    "GifInfo": "pipeline",
    "make_gif": "pipeline",
    "Splat": "splat",
    "predict": "splat",
    "save_gif": "gif",
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
