"""Synthetic scenes for tests that must not need the 2.8 GB weights."""

from __future__ import annotations

import numpy as np
import pytest

from splatgif.splat import Splat


def isotropic_cov(sigma: np.ndarray) -> np.ndarray:
    """(N,) world-space standard deviations -> (N, 6) covariance rows."""
    var = (np.asarray(sigma, np.float32) ** 2)[:, None]
    return var * np.array([1, 0, 0, 1, 0, 1], np.float32)


def photo_plane(rgb: np.ndarray, f_px: float, depth: float = 2.0, sigma_px: float = 0.7) -> Splat:
    """A fronto-parallel sheet of Gaussians, one per pixel, that looks like ``rgb``."""
    h, w, _ = rgb.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u.ravel() + 0.5 - w / 2) * depth / f_px
    y = (v.ravel() + 0.5 - h / 2) * depth / f_px
    n = h * w
    return Splat(
        means=np.stack([x, y, np.full(n, depth)], axis=1).astype(np.float32),
        cov=isotropic_cov(np.full(n, sigma_px * depth / f_px)),
        colors=(rgb.reshape(-1, 3) / 255.0).astype(np.float32),
        opacity=np.full(n, 0.99, np.float32),
        f_px=f_px,
        width=w,
        height=h,
    )


def dots(
    positions: list[tuple[float, float, float]], f_px: float, w: int, h: int, sigma_px=1.5
) -> Splat:
    """White Gaussian dots at camera-space (x, y, z) positions."""
    p = np.array(positions, np.float32)
    n = len(p)
    return Splat(
        means=p,
        cov=isotropic_cov(sigma_px * p[:, 2] / f_px),
        colors=np.ones((n, 3), np.float32),
        opacity=np.full(n, 0.99, np.float32),
        f_px=f_px,
        width=w,
        height=h,
    )


@pytest.fixture
def smooth_image() -> np.ndarray:
    """A smooth 64x48 colour image (gradients plus a soft blob): easy to reproduce with splats."""
    h, w = 48, 64
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 40 + 170 * u / w
    g = 40 + 170 * v / h
    b = 60 + 120 * np.exp(-(((u - 32) / 14) ** 2 + ((v - 24) / 10) ** 2))
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)
