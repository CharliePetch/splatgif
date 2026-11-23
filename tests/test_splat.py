import numpy as np
import pytest

from splatgif.splat import Splat

from .conftest import isotropic_cov


def two_depth_scene(f=100.0, w=200, h=100) -> Splat:
    """4,000 Gaussians at depth 2 in the middle of the frame, 4,000 at depth 5 in the left third."""
    rng = np.random.default_rng(1)
    n = 4000

    def sheet(u0, u1, depth):
        u, v = rng.uniform(u0, u1, n), rng.uniform(h * 0.3, h * 0.7, n)
        return np.stack([(u - w / 2) * depth / f, (v - h / 2) * depth / f, np.full(n, depth)], 1)

    means = np.concatenate([sheet(w * 0.45, w * 0.55, 2.0), sheet(w * 0.05, w * 0.25, 5.0)]).astype(
        np.float32
    )
    return Splat(
        means, isotropic_cov(np.full(2 * n, 0.01)), np.ones((2 * n, 3), np.float32),
        np.full(2 * n, 0.9, np.float32), f, w, h,
    )  # fmt: skip


def test_default_focus_is_the_depth_at_the_centre():
    pivot = two_depth_scene().focus_point(None)
    assert pivot == pytest.approx([0.0, 0.0, 2.0])


def test_focus_follows_the_chosen_pixel_along_its_ray():
    splat = two_depth_scene()
    pivot = splat.focus_point((0.15, 0.5))  # pixel (30, 50), in the far sheet
    assert pivot[2] == pytest.approx(5.0)
    assert pivot[0] == pytest.approx((30 - 100) * 5.0 / 100.0)  # x = (u - cx) * z / f
    assert pivot[1] == pytest.approx(0.0)


def test_sparse_window_falls_back_to_the_whole_scene():
    pivot = two_depth_scene().focus_point((0.99, 0.99))  # nothing projects there
    assert pivot[:2].tolist() == [0.0, 0.0]
    assert 2.0 <= pivot[2] <= 5.0
