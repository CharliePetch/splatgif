"""Depth maps: the scene's shape as one number per pixel.

SHARP predicts metric depth (it is Depth Pro underneath), so the numbers are metres, as good as
the focal length the photo gave it. A photo without EXIF lens data is read as a 30 mm lens and its
scale is then only approximate; ``DepthMap.exif_focal`` and the files record which case you have.
Chat apps usually strip EXIF, so expect ``False`` for photos that arrived that way.

How a pixel gets its number: the colour rasteriser (:mod:`splatgif.raster`) already sorts every
Gaussian nearest-first and bins them into tiles. The depth pass walks the very same lists with the
very same alpha maths, but instead of accumulating colour it watches opacity build up along the
pixel's ray and records the camera-space z of the Gaussian that carries it past one half: the
*median* depth of the ray. That is the surface the photograph shows at that pixel (frame 0 of the
GIF is the same Gaussians blended in the same order), and it stays sharp at silhouettes, where an
average would smear near and far together. It is z-depth (distance along the optical axis), not
distance from the camera centre. A pixel the Gaussians never make half-opaque is NaN.

Files, by suffix:

* ``.npz``: lossless float32 ``depth`` (NaN = no surface), plus ``f_px`` and ``exif_focal``; with
  ``cmap=True`` also a uint8 ``preview`` picture.
* ``.png`` / ``.tif``: a 16-bit grayscale map in the KITTI convention, value = metres x 256 and
  0 = no surface, good for 0.004 m to 255 m in 0.004 m steps (anything farther saturates). PNGs
  carry the lens info in a ``splatgif-depth`` text chunk, so :func:`load_depth` gives it back.
* ``cmap=True``, or any other suffix: a coloured picture for eyes only (red near, violet far,
  grey no surface), not a round trip.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numba import get_num_threads, njit, prange
from PIL import Image, PngImagePlugin

from .motion import frame_size
from .raster import ALPHA_MAX, TILE, _bin, _project

if TYPE_CHECKING:
    from .splat import Splat

SCALE = 256.0  # 16-bit maps store metres x 256 (KITTI's convention); 0 is "no surface"
OPAQUE = 0.5  # the depth of a pixel is where its ray's accumulated opacity first reaches this
_META_KEY = "splatgif-depth"
_PICTURE_ONLY = {".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# Near -> far, linear interpolation between stops. Red-ish reads as "close".
_CMAP_STOPS = np.array(
    [
        (224, 40, 32),
        (252, 140, 24),
        (246, 216, 64),
        (120, 212, 80),
        (44, 168, 152),
        (36, 116, 208),
        (68, 52, 160),
        (128, 32, 128),
    ],
    np.float64,
)


@dataclass(frozen=True)
class DepthMap:
    """Camera-space z for every pixel of the photo's own view.

    ``depth`` is (h, w) float32, metres if the lens was right; NaN where nothing opaque enough
    lies in front of the camera. ``f_px`` is the focal length of *this* pixel grid, so pixel
    (u, v) sits at ``((u + .5 - w/2) * z / f_px, (v + .5 - h/2) * z / f_px, z)`` in camera space.
    ``exif_focal`` is False when the photo had no lens data and 30 mm was assumed (the metres are
    then approximate), None when unknown (a file from elsewhere).
    """

    depth: np.ndarray
    f_px: float = float("nan")
    exif_focal: bool | None = None

    @property
    def height(self) -> int:
        return self.depth.shape[0]

    @property
    def width(self) -> int:
        return self.depth.shape[1]

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.depth)


@njit(parallel=True, cache=True)
def _depth_pass(packed, box, depth, offsets, ids, tile_order, tiles_x, w, h):
    """The colour blend's walk, recording a z instead of a colour.

    One tile per task, Gaussian-major inside the tile, nearest first: each Gaussian adds its
    alpha to the pixels it reaches, and the pixel keeps the z of the Gaussian that takes its
    accumulated opacity to OPAQUE. Every pixel is written only by its own tile's task, so there
    is nothing to race on, and the answer cannot depend on tile or thread order.
    """
    out = np.full((h, w), np.nan, np.float32)
    t_stop = 1.0 - OPAQUE
    for q in prange(tile_order.size):
        tile = tile_order[q]
        start, end = offsets[tile], offsets[tile + 1]
        if start == end:
            continue
        tx0 = (tile % tiles_x) * TILE
        ty0 = (tile // tiles_x) * TILE
        tx1 = min(tx0 + TILE, w) - 1
        ty1 = min(ty0 + TILE, h) - 1
        trans = np.ones((TILE, TILE), np.float32)  # light still passing through each pixel
        live = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)  # pixels not yet half opaque
        for m in range(start, end):
            i = ids[m]
            # The conic as _project stored it: xx, xy, yy (named a, b, c as in _blend).
            u, v, a, b, c, o = (
                packed[i, 0],
                packed[i, 1],
                packed[i, 2],
                packed[i, 3],
                packed[i, 4],
                packed[i, 5],
            )
            cutoff = packed[i, 9]
            z = depth[i]
            for py in range(max(ty0, box[i, 1]), min(ty1, box[i, 3]) + 1):
                dy = v - (py + 0.5)
                for px in range(max(tx0, box[i, 0]), min(tx1, box[i, 2]) + 1):
                    t = trans[py - ty0, px - tx0]
                    if t <= t_stop:
                        continue
                    dx = u - (px + 0.5)
                    power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
                    if power < cutoff:  # under one 8-bit level here
                        continue
                    t_next = t * (1.0 - min(o * math.exp(power), ALPHA_MAX))
                    trans[py - ty0, px - tx0] = t_next
                    if t_next <= t_stop:
                        out[py, px] = z
                        live -= 1
            if live == 0:
                break
    return out


def _f32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, np.float32)


def depth_map(splat: Splat, size: int = 640) -> DepthMap:
    """The scene's depth from the photo's own camera at rest.

    ``size`` is the long edge in pixels, as for the GIF; use the photo's own long edge to line
    the map up with it pixel for pixel. Frame 0 of ``splat.render`` at the same size matches it.
    """
    w, h = frame_size(size, splat.width / splat.height)
    fx, fy = splat.f_px * w / splat.width, splat.f_px * h / splat.height

    depth, packed, box = _project(
        _f32(splat.means), _f32(splat.cov), _f32(splat.colors), _f32(splat.opacity),
        np.eye(3), np.zeros(3), fx, fy, w / 2.0, h / 2.0, w, h,
    )  # fmt: skip
    visible = np.flatnonzero(box[:, 0] >= 0)
    if visible.size == 0:
        raise RuntimeError("nothing to measure: no Gaussian lands in front of the camera")
    order = visible[np.argsort(depth[visible])]  # nearest first

    tiles_x, tiles_y = -(-w // TILE), -(-h // TILE)
    offsets, ids = _bin(box, order, tiles_x, tiles_y, get_num_threads())
    tile_order = np.random.default_rng(0).permutation(tiles_x * tiles_y)  # as render_frame does
    z = _depth_pass(packed, box, depth, offsets, ids, tile_order, tiles_x, w, h)
    return DepthMap(depth=z, f_px=float(fx), exif_focal=splat.exif_focal)


def _ramp(n: int) -> np.ndarray:
    pos = np.linspace(0.0, 1.0, n) * (len(_CMAP_STOPS) - 1)
    lo = np.clip(pos.astype(int), 0, len(_CMAP_STOPS) - 2)
    frac = (pos - lo)[:, None]
    return np.round(_CMAP_STOPS[lo] + (_CMAP_STOPS[lo + 1] - _CMAP_STOPS[lo]) * frac).astype(
        np.uint8
    )


def preview(z: np.ndarray) -> np.ndarray:
    """(h, w, 3) uint8 picture of a depth array: red near, violet far, grey no surface.

    The ramp runs over *inverse* depth between the 2nd and 98th percentiles, which spends the
    colours on the foreground (where the detail is) instead of on a few distant outliers.
    """
    ok = np.isfinite(z)
    rgb = np.full(z.shape + (3,), 128, np.uint8)
    if not ok.any():
        return rgb
    inv = 1.0 / np.maximum(z[ok], 1e-6)
    lo, hi = np.percentile(inv, [2, 98])
    s = np.clip((hi - inv) / max(hi - lo, 1e-9), 0.0, 1.0)
    rgb[ok] = _ramp(256)[(s * 255).astype(np.int64)]
    return rgb


def _to_u16(z: np.ndarray) -> np.ndarray:
    ok = np.isfinite(z)
    return np.where(ok, np.clip(np.round(z * SCALE), 1, 65535), 0).astype(np.uint16)


def write_depth(dm: DepthMap, path: str | Path, *, cmap: bool = False) -> int:
    """Write ``dm`` to ``path`` (see the module docstring for the formats); returns the bytes."""
    z = np.asarray(dm.depth, np.float32)
    if z.ndim != 2:
        raise ValueError("a depth map must be a 2-D array")
    if not np.isfinite(z).any():
        raise RuntimeError("nothing to write: the whole depth map is empty")
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        arrays = {
            "depth": z,
            "f_px": np.float64(dm.f_px),
            "exif_focal": np.int8(-1 if dm.exif_focal is None else int(dm.exif_focal)),
        }
        if cmap:
            arrays["preview"] = preview(z)
        np.savez_compressed(path, **arrays)
    elif cmap or suffix in _PICTURE_ONLY or suffix not in {".png", ".tif", ".tiff"}:
        Image.fromarray(preview(z)).save(path)
    else:
        meta = PngImagePlugin.PngInfo()
        meta.add_text(
            _META_KEY,
            json.dumps(
                {"scale": SCALE, "unit": "m", "f_px": dm.f_px, "exif_focal": dm.exif_focal},
                allow_nan=True,
            ),
        )
        image = Image.fromarray(_to_u16(z))
        if suffix == ".png":
            image.save(path, pnginfo=meta)
        else:
            image.save(path)
    return path.stat().st_size


def load_depth(path: str | Path) -> DepthMap:
    """Read back a ``.npz`` or 16-bit ``.png`` / ``.tif`` written by :func:`write_depth`.

    The PNG comes back to within half a step (0.002 m); the ``.npz`` exactly. A coloured
    picture holds no numbers and is refused.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as f:
            flag = int(f["exif_focal"]) if "exif_focal" in f else -1
            return DepthMap(
                depth=np.asarray(f["depth"], np.float32),
                f_px=float(f["f_px"]) if "f_px" in f else float("nan"),
                exif_focal=None if flag < 0 else bool(flag),
            )
    with Image.open(path) as img:
        if img.mode not in ("I;16", "I"):
            raise ValueError(f"{path}: not a splatgif depth file (need .npz or a 16-bit map)")
        raw = np.asarray(img).astype(np.float32)
        meta = json.loads(img.info.get(_META_KEY, "{}"))
    z = raw / float(meta.get("scale", SCALE))
    z[raw == 0] = np.nan
    return DepthMap(
        depth=z,
        f_px=float(meta.get("f_px") if meta.get("f_px") is not None else float("nan")),
        exif_focal=meta.get("exif_focal"),
    )
