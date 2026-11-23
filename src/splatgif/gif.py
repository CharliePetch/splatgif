"""Writing the frames out as one looping GIF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, features

_PALETTE_SAMPLES = 300_000  # pixels, spread over every frame, that the palette is built from


def _palette_image(frames: list[Image.Image]) -> Image.Image:
    """One 256-colour palette for the whole loop.

    Built from pixels sampled across *all* frames: a palette per frame makes colours crawl from
    frame to frame, and one taken from frame 0 alone misses colours that only appear mid-sway.
    """
    total = sum(f.width * f.height for f in frames)
    stride = max(1, total // _PALETTE_SAMPLES)
    sample = np.concatenate([np.asarray(f).reshape(-1, 3)[::stride] for f in frames])
    method = (
        Image.Quantize.LIBIMAGEQUANT
        if features.check("libimagequant")
        else Image.Quantize.MEDIANCUT
    )
    return Image.fromarray(sample[None]).quantize(256, method=method)


def save_gif(
    frames: list[Image.Image], path: str | Path, delay_ms: int = 20, dither: bool = True
) -> int:
    """Write ``frames`` as an endlessly looping GIF; returns the size in bytes.

    ``delay_ms`` is per frame; GIF counts in hundredths of a second, so it is rounded to 10 ms.
    Dithering (Floyd-Steinberg) trades the banding that 256 colours cause on smooth gradients for
    fine grain, at the cost of a larger file.
    """
    if not frames:
        raise ValueError("no frames to write")
    palette = _palette_image(frames)
    mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    indexed = [f.quantize(palette=palette, dither=mode) for f in frames]
    path = Path(path)
    indexed[0].save(
        path,
        save_all=True,
        append_images=indexed[1:],
        duration=delay_ms,
        loop=0,
        optimize=False,
    )
    return path.stat().st_size
