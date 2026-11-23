"""Stand-ins for the SHARP network and other synthetic fixtures (no weights needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from splatgif.model import Raw
from splatgif.splat import Splat


class FakePredictor:
    """Describes the photo as a flat sheet of Gaussians at depth 3, in the network's NDC terms."""

    def __call__(self, image, f_px, width):
        g, z = 48, 3.0
        colors = F.interpolate(image, size=(g, g), mode="area")[0]
        ndc = 2 * (torch.arange(g) + 0.5) / g - 1
        gy, gx = torch.meshgrid(ndc, ndc, indexing="ij")
        n = g * g
        return Raw(
            means=torch.stack(
                [z * gx.reshape(-1), z * gy.reshape(-1), torch.full((n,), z)], dim=-1
            ),
            scales=torch.full((n, 3), 0.6 * z * 2 / g),
            quats=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
            colors=colors.reshape(3, -1).T.contiguous(),
            opacities=torch.full((n,), 0.9),
        )


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
def fake_model(monkeypatch):
    """Make splatgif.predict() cheap and exact: a flat sheet at depth 3, no weights needed."""
    from splatgif import splat

    monkeypatch.setattr(splat, "resolve_weights", lambda *a, **k: "unused.pt")
    monkeypatch.setattr(splat, "load_predictor", lambda *a, **k: FakePredictor())
    return FakePredictor


@pytest.fixture
def smooth_image() -> np.ndarray:
    """A smooth 64x48 colour image (gradients plus a soft blob): easy to reproduce with splats."""
    h, w = 48, 64
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 40 + 170 * u / w
    g = 40 + 170 * v / h
    b = 60 + 120 * np.exp(-(((u - 32) / 14) ** 2 + ((v - 24) / 10) ** 2))
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


@pytest.fixture
def photo_file(tmp_path, smooth_image):
    """The smooth image on disk, 96 x 64: landscape, so x and y scales of the camera differ."""
    path = tmp_path / "scene.jpg"
    Image.fromarray(smooth_image).resize((96, 64), Image.Resampling.BILINEAR).save(path, quality=98)
    return path
