import math

import numpy as np
import pytest

from splatgif.motion import frame_size, peak_degrees, sway
from splatgif.raster import render_frame
from splatgif.splat import Splat

from .conftest import dots, photo_plane


def centroid(frame: np.ndarray, region: tuple[slice, slice]) -> tuple[float, float]:
    """Brightness-weighted (x, y) centre of the dot inside ``region`` of a (h, w, 3) frame."""
    ys, xs = region
    patch = frame[ys, xs].astype(np.float64).sum(axis=-1)
    gy, gx = np.mgrid[ys, xs]
    return float((patch * gx).sum() / patch.sum() + 0.5), float(
        (patch * gy).sum() / patch.sum() + 0.5
    )


def test_frame_zero_is_the_photograph(smooth_image):
    h, w, _ = smooth_image.shape
    splat = photo_plane(smooth_image, f_px=80.0)
    frame = render_frame(splat, 0.0, np.array([0.0, 0.0, 2.0]), w, h, 80.0, 80.0)
    err = np.abs(frame.astype(int) - smooth_image.astype(int))
    assert err.mean() < 3.0
    assert err.max() < 20


def test_loop_closes_and_starts_on_the_photo():
    angles = sway("moderate", 48)
    assert angles[0] == 0.0
    assert abs(angles[24]) < 1e-12  # half way round: back at the photo's own viewpoint
    assert np.isclose(angles.max(), math.radians(2.5), rtol=1e-3)
    assert np.isclose(angles.min(), -math.radians(2.5), rtol=1e-3)
    assert not np.isclose(angles[-1], angles[0])  # no duplicated closing frame


def test_motion_presets_and_degrees():
    assert peak_degrees("drastic") == 8.0
    assert peak_degrees(4) == 4.0
    with pytest.raises(ValueError):
        peak_degrees("wild")


@pytest.mark.parametrize(
    ("long_edge", "aspect", "expected"),
    [
        (640, 4 / 3, (640, 480)),
        (1280, 3 / 4, (960, 1280)),
        (101, 1.0, (100, 100)),
        (640, 0.7501, (480, 640)),
    ],
)
def test_frame_size_is_even(long_edge, aspect, expected):
    w, h = frame_size(long_edge, aspect)
    assert (w, h) == expected
    assert w % 2 == 0 and h % 2 == 0


def test_parallax_pins_the_focus_and_splits_near_from_far():
    f, w, h = 100.0, 160, 96
    focus, near, far = (0.0, 0.0, 1.0), (0.2, 0.0, 0.5), (-0.4, 0.0, 2.0)
    # Same row, well apart, so each dot has its own box to be measured in.
    splat = dots([focus, near, far], f, w, h)
    pivot = np.array(focus)
    yaw = 0.08

    def where(p, angle):
        c, s = math.cos(angle), math.sin(angle)
        rot = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        q = pivot + rot @ (np.array(p) - pivot)
        return f * q[0] / q[2] + w / 2, f * q[1] / q[2] + h / 2

    still = render_frame(splat, 0.0, pivot, w, h, f, f)
    moved = render_frame(splat, yaw, pivot, w, h, f, f)
    for p in (focus, near, far):
        u0, v0 = where(p, 0.0)
        u1, v1 = where(p, yaw)
        box = (slice(int(v0) - 8, int(v0) + 9), slice(int(min(u0, u1)) - 14, int(max(u0, u1)) + 15))
        cx0, cy0 = centroid(still, box)
        cx1, cy1 = centroid(moved, box)
        assert (cx0, cy0) == pytest.approx((u0, v0), abs=0.35)
        assert (cx1, cy1) == pytest.approx((u1, v1), abs=0.35)

    shift = {name: where(p, yaw)[0] - where(p, 0.0)[0] for name, p in
             [("focus", focus), ("near", near), ("far", far)]}  # fmt: skip
    assert abs(shift["focus"]) < 1e-6  # the pivot does not move
    assert shift["near"] > 1.0 and shift["far"] < -1.0  # opposite directions about the pivot


def test_render_returns_one_frame_per_step(smooth_image):
    h, w, _ = smooth_image.shape
    splat = photo_plane(smooth_image, f_px=80.0)
    frames = splat.render(size=64, motion="drastic", frames=8)
    assert len(frames) == 8
    assert all(f.size == (64, 48) for f in frames)
    # Half way round the loop the camera is back where it started.
    assert np.abs(np.asarray(frames[4]).astype(int) - np.asarray(frames[0]).astype(int)).max() <= 1
    # ... and a quarter way round it has genuinely moved.
    assert (
        np.abs(np.asarray(frames[2]).astype(int) - np.asarray(frames[0]).astype(int)).mean() > 0.05
    )


def test_empty_view_is_black_not_an_error():
    splat = Splat(
        means=np.array([[0, 0, -5.0]], np.float32),  # behind the camera
        cov=np.array([[1e-4, 0, 0, 1e-4, 0, 1e-4]], np.float32),
        colors=np.ones((1, 3), np.float32),
        opacity=np.array([0.9], np.float32),
        f_px=50.0, width=32, height=32,
    )  # fmt: skip
    frame = render_frame(splat, 0.0, np.array([0.0, 0.0, 1.0]), 32, 32, 50.0, 50.0)
    assert frame.shape == (32, 32, 3) and frame.max() == 0


@pytest.mark.parametrize(("near", "far"), [((0, 0, 1), (1, 0, 0)), ((1, 0, 0), (0, 0, 1))])
def test_nearer_surface_hides_farther_one(near, far):
    f, w, h = 60.0, 32, 32
    # Listed far-first and near-first in turn: the result must follow depth, not list order.
    for order in (1, -1):
        colors = np.array([far, near], np.float32)[::order]
        means = np.array([[0, 0, 3.0], [0, 0, 1.0]], np.float32)[::order]
        splat = Splat(
            means, np.array([[4e-3, 0, 0, 4e-3, 0, 4e-3]] * 2, np.float32), colors,
            np.array([0.99, 0.99], np.float32), f, w, h,
        )  # fmt: skip
        px = render_frame(splat, 0.0, np.array([0.0, 0.0, 1.0]), w, h, f, f)[h // 2, w // 2].astype(
            int
        )
        winner, loser = (px[2], px[0]) if near == (0, 0, 1) else (px[0], px[2])
        assert winner > 200 and loser < 60
