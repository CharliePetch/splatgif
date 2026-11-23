"""End to end with the real checkpoint: `pytest -m slow` (2.8 GB of weights, a few CPU minutes)."""

import numpy as np
import pytest
from PIL import Image

import splatgif
from splatgif import weights

pytestmark = pytest.mark.slow

if not weights.default_path().is_file():
    pytest.skip("SHARP weights are not cached; run splatgif once first", allow_module_level=True)


def synthetic_photo(path, w=640, h=480):
    """Sky, a ground plane and two blocks: enough structure for SHARP to find depth in."""
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.float64)
    img[:] = [120, 170, 230]  # sky
    ground = y > h * 0.6
    img[ground] = np.stack([90 + 60 * (y[ground] - h * 0.6) / (h * 0.4)] * 3, axis=-1) * [
        1.0,
        1.1,
        0.7,
    ]
    img[(abs(x - w * 0.3) < 60) & (abs(y - h * 0.62) < 70)] = [200, 60, 50]
    img[(abs(x - w * 0.68) < 40) & (abs(y - h * 0.66) < 45)] = [50, 90, 200]
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path, quality=95)


def test_photo_to_gif_with_the_real_model(tmp_path):
    photo = tmp_path / "scene.jpg"
    synthetic_photo(photo)
    info = splatgif.make_gif(photo, tmp_path / "out.gif", size=320, frames=8, device="cpu")

    assert info.path.stat().st_size > 10_000
    assert 0.9e6 < info.gaussians <= 1_179_648
    assert info.focus_depth > 0 and not info.exif_focal
    with Image.open(info.path) as gif:
        assert gif.size == (320, 240) and gif.info["loop"] == 0


def test_depth_map_of_the_real_scene(tmp_path):
    photo = tmp_path / "scene.jpg"
    synthetic_photo(photo)
    out = tmp_path / "d.npz"
    info = splatgif.make_depth(photo, out, size=320, device="cpu")

    assert (info.width, info.height) == (320, 240) and not info.exif_focal
    assert 0.2 < info.min_depth < info.max_depth < 255.0
    assert info.invalid_px < 0.1 * 320 * 240
    dm = splatgif.load_depth(out)
    assert (dm.width, dm.height) == (320, 240) and dm.exif_focal is False
    # A ground plane runs away from the camera up the frame: its bottom rows are nearer than
    # the rows just under the horizon.
    near_ground = np.nanmedian(dm.depth[-8:, :])
    far_ground = np.nanmedian(dm.depth[int(240 * 0.62) : int(240 * 0.66), :])
    assert near_ground < far_ground
