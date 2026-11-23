"""photo -> GIF in one call."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .gif import save_gif
from .photo import load_photo
from .splat import pick_device, predict


@dataclass(frozen=True)
class GifInfo:
    path: Path
    width: int
    height: int
    frames: int
    delay_ms: int
    bytes: int
    focal_px: float
    exif_focal: bool  # False: no focal length in the photo, so 30 mm was assumed
    focus_depth: float  # distance to the point the sway pivots on, in SHARP's units
    gaussians: int
    device: str
    predict_s: float
    render_s: float


def make_gif(
    image: str | Path,
    output: str | Path | None = None,
    *,
    size: int = 640,
    motion: str | float = "moderate",
    frames: int = 48,
    delay_ms: int = 20,
    focus: tuple[float, float] | None = None,
    dither: bool = True,
    device: str | None = None,
    weights: str | Path | None = None,
    accept_license: bool = False,
    log: Callable[[str], None] | None = None,
) -> GifInfo:
    """Turn one photo into a looping 3D "wigglegram" GIF and return what was made.

    ``size`` is the long edge in pixels. ``motion`` is ``"moderate"`` (about 2.5 degrees either
    side, like a Nishika film camera), ``"drastic"`` (8 degrees), or the peak yaw in degrees.
    ``focus`` is the pixel the sway pivots on, as (x, y) fractions of the frame; it stays put
    while everything nearer or farther drifts. Frame 0 is always the photograph itself.
    """
    say = log or (lambda _msg: None)
    image = Path(image)
    output = Path(output) if output else image.with_suffix(".gif")

    photo = load_photo(image)
    note = "" if photo.exif_focal else "  (no EXIF focal length: assuming a 30 mm lens)"
    say(f"{image.name}: {photo.width}x{photo.height}, f = {photo.f_px:.0f} px{note}")

    dev = pick_device(device)
    say(f"predicting a Gaussian splat with SHARP on {dev.type} ...")
    started = time.perf_counter()
    splat = predict(photo, weights=weights, device=str(dev), accept_license=accept_license)
    predict_s = time.perf_counter() - started
    say(f"  {len(splat.means):,} Gaussians in {predict_s:.0f} s")

    pivot = splat.focus_point(focus)
    say(f"rendering {frames} frames, pivot depth {pivot[2]:.3g} ...")
    started = time.perf_counter()
    rendered = splat.render(size=size, motion=motion, frames=frames, focus=focus)
    nbytes = save_gif(rendered, output, delay_ms=delay_ms, dither=dither)
    render_s = time.perf_counter() - started
    say(f"wrote {output} ({nbytes / 1e6:.1f} MB) in {render_s:.0f} s")

    return GifInfo(
        path=output,
        width=rendered[0].width,
        height=rendered[0].height,
        frames=len(rendered),
        delay_ms=delay_ms,
        bytes=nbytes,
        focal_px=photo.f_px,
        exif_focal=photo.exif_focal,
        focus_depth=float(pivot[2]),
        gaussians=len(splat.means),
        device=dev.type,
        predict_s=predict_s,
        render_s=render_s,
    )
