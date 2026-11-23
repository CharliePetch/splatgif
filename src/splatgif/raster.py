"""A small CPU Gaussian-splat rasteriser.

Standard 3D Gaussian Splatting, compiled with numba so it needs no GPU and no compiler at install
time: EWA projection of each Gaussian to a 2D ellipse, depth sort, 16x16 tile binning, then
front-to-back alpha blending with early exit. The background is black.

Colours are blended as sRGB values, the way public splat viewers do (and the way the browser
pipeline this replaces did). Apple's own renderer blends in linear light; the two differ by a few
8-bit levels where Gaussians only partly cover a pixel, and sRGB blending showed slightly less of
the sky speckle that thin foliage splats produce.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numba import get_num_threads, njit, prange

if TYPE_CHECKING:
    from .splat import Splat

TILE = 16
LOW_PASS = 0.075  # px^2 added to the 2D covariance diagonal (3DGS uses 0.3, which blurs SHARP's
# small splats; 0.075 keeps the photo's sharpness and still tames sub-pixel speckle)
NEAR = 0.01
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
T_MIN = 1e-4  # stop blending a pixel once less than this much light gets through


@njit(parallel=True, fastmath=True, cache=True)
def _project(means, cov, rgb, opacity, rot, shift, fx, fy, cx, cy, w, h):
    """Per Gaussian: depth, screen-space ellipse, colour and pixel bounding box."""
    n = means.shape[0]
    depth = np.empty(n, np.float32)
    packed = np.empty((n, 10), np.float32)  # u, v, conic a, b, c, opacity, r, g, b, cutoff
    box = np.full((n, 4), -1, np.int32)  # x0, y0, x1, y1 in pixels; x0 < 0: culled
    lim_x = 1.3 * 0.5 * w / fx  # keep the Jacobian sane for Gaussians far off-axis
    lim_y = 1.3 * 0.5 * h / fy
    for i in prange(n):
        px, py, pz = means[i, 0], means[i, 1], means[i, 2]
        x = rot[0, 0] * px + rot[0, 1] * py + rot[0, 2] * pz + shift[0]
        y = rot[1, 0] * px + rot[1, 1] * py + rot[1, 2] * pz + shift[1]
        z = rot[2, 0] * px + rot[2, 1] * py + rot[2, 2] * pz + shift[2]
        depth[i] = z
        if z < NEAR:
            continue

        # Sigma' = R Sigma R^T, kept as six numbers.
        c00, c01, c02, c11, c12, c22 = (
            cov[i, 0],
            cov[i, 1],
            cov[i, 2],
            cov[i, 3],
            cov[i, 4],
            cov[i, 5],
        )
        m00 = rot[0, 0] * c00 + rot[0, 1] * c01 + rot[0, 2] * c02
        m01 = rot[0, 0] * c01 + rot[0, 1] * c11 + rot[0, 2] * c12
        m02 = rot[0, 0] * c02 + rot[0, 1] * c12 + rot[0, 2] * c22
        m10 = rot[1, 0] * c00 + rot[1, 1] * c01 + rot[1, 2] * c02
        m11 = rot[1, 0] * c01 + rot[1, 1] * c11 + rot[1, 2] * c12
        m12 = rot[1, 0] * c02 + rot[1, 1] * c12 + rot[1, 2] * c22
        m20 = rot[2, 0] * c00 + rot[2, 1] * c01 + rot[2, 2] * c02
        m21 = rot[2, 0] * c01 + rot[2, 1] * c11 + rot[2, 2] * c12
        m22 = rot[2, 0] * c02 + rot[2, 1] * c12 + rot[2, 2] * c22
        s00 = m00 * rot[0, 0] + m01 * rot[0, 1] + m02 * rot[0, 2]
        s01 = m00 * rot[1, 0] + m01 * rot[1, 1] + m02 * rot[1, 2]
        s02 = m00 * rot[2, 0] + m01 * rot[2, 1] + m02 * rot[2, 2]
        s11 = m10 * rot[1, 0] + m11 * rot[1, 1] + m12 * rot[1, 2]
        s12 = m10 * rot[2, 0] + m11 * rot[2, 1] + m12 * rot[2, 2]
        s22 = m20 * rot[2, 0] + m21 * rot[2, 1] + m22 * rot[2, 2]

        # EWA: 2D covariance = J Sigma' J^T for the pinhole Jacobian J.
        tx = min(lim_x, max(-lim_x, x / z))
        ty = min(lim_y, max(-lim_y, y / z))
        j00, j02 = fx / z, -fx * tx / z
        j11, j12 = fy / z, -fy * ty / z
        a = j00 * j00 * s00 + 2.0 * j00 * j02 * s02 + j02 * j02 * s22 + LOW_PASS
        b = j00 * j11 * s01 + j00 * j12 * s02 + j02 * j11 * s12 + j02 * j12 * s22
        c = j11 * j11 * s11 + 2.0 * j11 * j12 * s12 + j12 * j12 * s22 + LOW_PASS
        det = a * c - b * b
        if det <= 0.0:
            continue

        # How far out does this Gaussian still add at least one 8-bit level of colour?
        o = opacity[i]
        if o < ALPHA_MIN:
            continue
        k = math.sqrt(2.0 * math.log(255.0 * o))
        rx, ry = k * math.sqrt(a), k * math.sqrt(c)
        u = fx * x / z + cx
        v = fy * y / z + cy
        x0 = max(0, int(math.ceil(u - rx - 0.5)))
        x1 = min(w - 1, int(math.floor(u + rx - 0.5)))
        y0 = max(0, int(math.ceil(v - ry - 0.5)))
        y1 = min(h - 1, int(math.floor(v + ry - 0.5)))
        if x0 > x1 or y0 > y1:
            continue

        inv = 1.0 / det
        packed[i, 0], packed[i, 1] = u, v
        packed[i, 2], packed[i, 3], packed[i, 4] = c * inv, -b * inv, a * inv
        packed[i, 5] = o
        packed[i, 6], packed[i, 7], packed[i, 8] = rgb[i, 0], rgb[i, 1], rgb[i, 2]
        packed[i, 9] = -0.5 * k * k  # below this exponent the Gaussian adds under one 8-bit level
        box[i, 0], box[i, 1], box[i, 2], box[i, 3] = x0, y0, x1, y1
    return depth, packed, box


@njit(parallel=True, cache=True)
def _bin(box, order, tiles_x, tiles_y, chunks):
    """Per-tile lists of Gaussian indices, nearest first, as offsets into one flat array.

    A counting sort run in parallel over ``chunks`` contiguous slices of the depth order: within a
    tile the slices are laid out nearest-first, and each slice writes its own disjoint range.
    """
    n_tiles = tiles_x * tiles_y
    n = order.size
    lo = np.empty(chunks + 1, np.int64)
    for c in range(chunks + 1):
        lo[c] = n * c // chunks

    counts = np.zeros((chunks, n_tiles), np.int64)
    for c in prange(chunks):
        for k in range(lo[c], lo[c + 1]):
            i = order[k]
            for ty in range(box[i, 1] // TILE, box[i, 3] // TILE + 1):
                for tx in range(box[i, 0] // TILE, box[i, 2] // TILE + 1):
                    counts[c, ty * tiles_x + tx] += 1

    offsets = np.zeros(n_tiles + 1, np.int64)
    cursor = np.empty((chunks, n_tiles), np.int64)
    total = 0
    for t in range(n_tiles):
        offsets[t] = total
        for c in range(chunks):
            cursor[c, t] = total
            total += counts[c, t]
    offsets[n_tiles] = total

    ids = np.empty(total, np.int32)
    for c in prange(chunks):
        for k in range(lo[c], lo[c + 1]):
            i = order[k]
            for ty in range(box[i, 1] // TILE, box[i, 3] // TILE + 1):
                for tx in range(box[i, 0] // TILE, box[i, 2] // TILE + 1):
                    t = ty * tiles_x + tx
                    ids[cursor[c, t]] = i
                    cursor[c, t] += 1
    return offsets, ids


@njit(parallel=True, fastmath=True, cache=True)
def _blend(packed, box, offsets, ids, tile_order, tiles_x, w, h):
    """Front-to-back alpha blending, one tile per thread task.

    Gaussian-major inside a tile: each Gaussian walks only the pixels it can reach, so no time is
    spent testing the many Gaussians that merely overlap the tile.
    """
    out = np.zeros((h, w, 3), np.uint8)
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
        acc = np.zeros((TILE, TILE, 3), np.float32)
        live = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)  # pixels still taking colour
        for m in range(start, end):
            i = ids[m]
            u, v, a, b, c, o = (
                packed[i, 0],
                packed[i, 1],
                packed[i, 2],
                packed[i, 3],
                packed[i, 4],
                packed[i, 5],
            )
            cutoff = packed[i, 9]
            for py in range(max(ty0, box[i, 1]), min(ty1, box[i, 3]) + 1):
                dy = v - (py + 0.5)
                for px in range(max(tx0, box[i, 0]), min(tx1, box[i, 2]) + 1):
                    t = trans[py - ty0, px - tx0]
                    if t < T_MIN:
                        continue
                    dx = u - (px + 0.5)
                    power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
                    if power < cutoff:  # under one 8-bit level here: skip the exp
                        continue
                    alpha = min(o * math.exp(power), ALPHA_MAX)
                    t_next = t * (1.0 - alpha)
                    if t_next < T_MIN:  # this pixel is opaque now; leave it be
                        trans[py - ty0, px - tx0] = 0.0
                        live -= 1
                        continue
                    wgt = alpha * t
                    acc[py - ty0, px - tx0, 0] += packed[i, 6] * wgt
                    acc[py - ty0, px - tx0, 1] += packed[i, 7] * wgt
                    acc[py - ty0, px - tx0, 2] += packed[i, 8] * wgt
                    trans[py - ty0, px - tx0] = t_next
            if live == 0:
                break
        for py in range(ty0, ty1 + 1):
            for px in range(tx0, tx1 + 1):
                for ch in range(3):
                    out[py, px, ch] = int(min(255.0, acc[py - ty0, px - tx0, ch] * 255.0 + 0.5))
    return out


def render_frame(
    splat: Splat, yaw: float, pivot: np.ndarray, w: int, h: int, fx: float, fy: float
) -> np.ndarray:
    """One (h, w, 3) uint8 frame with the scene yawed by ``-yaw`` radians about ``pivot``.

    That is the photo camera swung round ``pivot`` by ``yaw`` (same intrinsics, still looking at
    it); ``yaw = 0`` reproduces the photograph.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], np.float64)
    shift = pivot - rot @ pivot

    f32 = lambda a: np.ascontiguousarray(a, np.float32)  # noqa: E731
    depth, packed, box = _project(
        f32(splat.means), f32(splat.cov), f32(splat.colors), f32(splat.opacity),
        rot, shift, fx, fy, w / 2.0, h / 2.0, w, h,
    )  # fmt: skip
    visible = np.flatnonzero(box[:, 0] >= 0)
    order = visible[np.argsort(depth[visible])]  # nearest first

    tiles_x, tiles_y = -(-w // TILE), -(-h // TILE)
    offsets, ids = _bin(box, order, tiles_x, tiles_y, get_num_threads())
    # Shuffled so every thread gets a mix of busy and empty tiles.
    tile_order = np.random.default_rng(0).permutation(tiles_x * tiles_y)
    return _blend(packed, box, offsets, ids, tile_order, tiles_x, w, h)
