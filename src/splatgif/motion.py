"""The camera sway: plain maths, nothing heavy to import."""

from __future__ import annotations

import numpy as np

# Peak yaw of the sway, in degrees, either side of the photo's own viewpoint. A real Nishika
# film camera works out near 1 degree, which reads as mush on a screen; 2.5 is a little more than
# life, which is what makes the effect legible. 8 is frankly exaggerated, still inside the cone a
# single-photo splat can support.
MOTIONS = {"moderate": 2.5, "drastic": 8.0}


def peak_degrees(motion: str | float) -> float:
    """A preset name or an explicit peak yaw in degrees -> degrees."""
    if isinstance(motion, str):
        try:
            return MOTIONS[motion]
        except KeyError:
            raise ValueError(
                f"unknown motion {motion!r}; use {', '.join(MOTIONS)} or degrees"
            ) from None
    return float(motion)


def sway(motion: str | float, frames: int) -> np.ndarray:
    """Yaw offsets in radians, one per frame: a sine that starts at 0 (the photograph itself).

    The loop never repeats its first frame at the end: a GIF returns to frame 0 on its own, so a
    closing duplicate would hold the same picture for two frames and put a hitch in the loop.
    """
    return np.deg2rad(peak_degrees(motion)) * np.sin(2 * np.pi * np.arange(frames) / frames)


def frame_size(long_edge: int, aspect: float) -> tuple[int, int]:
    """(width, height) in pixels for a long edge and a width / height aspect; both even."""

    def even(v: float) -> int:
        return max(2, round(v / 2) * 2)

    if aspect >= 1:
        return even(long_edge), even(long_edge / aspect)
    return even(long_edge * aspect), even(long_edge)
