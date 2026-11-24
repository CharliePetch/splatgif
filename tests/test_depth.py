"""Depth maps: geometry on synthetic scenes, the file formats, and the CLI mode (no weights)."""

import json

import numpy as np
import pytest
from PIL import Image

import splatgif
from splatgif import cli
from splatgif.depth import SCALE, DepthMap, depth_map, load_depth, preview, write_depth
from splatgif.splat import Splat

from .conftest import dots, isotropic_cov, photo_plane


def joined(*parts: Splat) -> Splat:
    first = parts[0]
    return Splat(
        np.concatenate([p.means for p in parts]),
        np.concatenate([p.cov for p in parts]),
        np.concatenate([p.colors for p in parts]),
        np.concatenate([p.opacity for p in parts]),
        first.f_px,
        first.width,
        first.height,
    )


def wall(f_px, w, h, depth, step=0.08, sigma_px=1.5) -> Splat:
    """A fronto-parallel layer of overlapping dots that covers the whole frame at ``depth``."""
    span = 1.4 * depth / 3.0
    xs = np.arange(-span, span + 1e-9, step * depth / 3.0)
    return dots([(x, y, depth) for x in xs for y in xs], f_px, w, h, sigma_px=sigma_px)


def test_flat_sheet_reads_its_own_depth(smooth_image):
    splat = photo_plane(smooth_image, f_px=80.0, depth=2.0)
    dm = depth_map(splat, size=64)
    assert dm.valid.all(), "one Gaussian per pixel leaves no holes, edges included"
    assert np.abs(dm.depth - 2.0).max() < 1e-4


def test_near_object_pops_off_the_back_wall():
    f, w, h = 100.0, 64, 64
    splat = joined(wall(f, w, h, 3.0), dots([(0.0, 0.0, 1.0)], f, w, h))
    z = depth_map(splat, 64).depth
    assert abs(z[32, 32] - 1.0) < 1e-4, "the bead's own depth must win at the centre"
    assert abs(z[32, 4] - 3.0) < 1e-4, "far from the bead the wall reads 3"


def test_depth_follows_the_geometry_not_the_list_order():
    rng = np.random.default_rng(7)
    f, w, h, n = 90.0, 64, 48, 400
    pos = np.stack(
        [rng.uniform(-0.45, 0.45, n), rng.uniform(-0.33, 0.33, n), rng.uniform(1.0, 3.0, n)], 1
    )
    base = Splat(
        pos.astype(np.float32),
        isotropic_cov(2.5 * pos[:, 2] / f),
        np.ones((n, 3), np.float32),
        np.full(n, 0.6, np.float32),
        f,
        w,
        h,
    )
    perm = rng.permutation(n)
    shuffled = Splat(
        base.means[perm], base.cov[perm], base.colors[perm], base.opacity[perm], f, w, h
    )
    a, b = depth_map(base, 64).depth, depth_map(shuffled, 64).depth
    assert np.isfinite(a).sum() > 200 and (np.isnan(a) == np.isnan(b)).all()
    assert np.array_equal(a[np.isfinite(a)], b[np.isfinite(b)])


def test_median_depth_is_where_the_ray_turns_half_opaque():
    # Two translucent layers on one ray: 0.3 at z=1 then 0.4 at z=2. After the first the ray is
    # 30% opaque, after the second 1 - 0.7 * 0.6 = 58%: the pixel belongs to the second layer.
    f = 50.0

    def layer(z, alpha):
        s = dots([(0.0, 0.0, z)], f, 16, 16, sigma_px=4.0)
        return Splat(s.means, s.cov, s.colors, np.array([alpha], np.float32), f, 16, 16)

    z = depth_map(joined(layer(1.0, 0.3), layer(2.0, 0.4)), 16).depth
    assert abs(z[8, 8] - 2.0) < 1e-4
    thin = depth_map(joined(layer(1.0, 0.3), layer(2.0, 0.1)), 16).depth
    assert np.isnan(thin[8, 8]), "30% then 37% opaque never reaches one half: a hole"
    thick = depth_map(joined(layer(1.0, 0.6), layer(2.0, 0.9)), 16).depth
    assert abs(thick[8, 8] - 1.0) < 1e-4, "already past one half at the first layer"


def test_silhouette_has_no_halo():
    f, w, h = 100.0, 64, 64
    splat = joined(wall(f, w, h, 3.0), dots([(0.0, 0.0, 1.0)], f, w, h, sigma_px=3.0))
    z = depth_map(splat, 64).depth
    near = z < 2.0
    # Pixels count as near only where the bead itself is at least half opaque: that is a disc
    # of radius sigma * sqrt(2 ln(0.99/0.5)) = 3.5 px, not the 3.1-sigma = 9 px the colour
    # pass's 1/255 cutoff would give.
    assert 25 <= near.sum() <= 60, near.sum()


def test_gaps_stay_holes_not_guesses():
    z = depth_map(dots([(0.0, 0.0, 1.0)], 60.0, 48, 48), 48).depth
    assert np.isnan(z[0, 0]) and np.isnan(z[47, 47])
    assert abs(z[24, 24] - 1.0) < 1e-4
    assert np.isfinite(z).sum() < 0.1 * z.size


def test_anisotropic_splats_use_the_right_axes():
    # A long thin horizontal ellipse: the map must reach farther along x than along y. Swapped
    # conic terms (xx <-> yy) would stretch it vertically instead, and isotropic tests miss that.
    f, w, h = 80.0, 64, 64
    cov = np.array([[(6 * 2.0 / f) ** 2, 0, 0, (1 * 2.0 / f) ** 2, 0, (0.01) ** 2]], np.float32)
    s = Splat(
        np.array([[0, 0, 2.0]], np.float32), cov, np.ones((1, 3), np.float32),
        np.array([0.99], np.float32), f, w, h,
    )  # fmt: skip
    ok = depth_map(s, 64).valid
    xs, ys = np.flatnonzero(ok.any(axis=0)), np.flatnonzero(ok.any(axis=1))
    assert xs.size > 2 * ys.size >= 4, (xs.size, ys.size)


def test_rotated_splats_lean_the_right_way():
    # Long axis along the +x,+y diagonal (xy covariance > 0, y down): the covered pixels must
    # correlate positively. A flipped sign on the cross term leans them the other way.
    f, w, h, z = 80.0, 64, 64, 2.0
    px = z / f  # world units per pixel at this depth
    long_, short = 6 * px, 1.5 * px
    xx = yy = (long_**2 + short**2) / 2
    xy = (long_**2 - short**2) / 2
    cov = np.array([[xx, xy, 0, yy, 0, 0.01**2]], np.float32)
    s = Splat(
        np.array([[0, 0, z]], np.float32), cov, np.ones((1, 3), np.float32),
        np.array([0.99], np.float32), f, w, h,
    )  # fmt: skip
    ys, xs = np.nonzero(depth_map(s, 64).valid)
    assert xs.size > 20 and np.corrcoef(xs, ys)[0, 1] > 0.5


def test_everything_behind_the_camera_is_a_clean_error():
    with pytest.raises(RuntimeError, match="nothing to measure"):
        depth_map(dots([(0.0, 0.0, -5.0)], 60.0, 32, 32), 32)


def test_frame_zero_and_depth_agree_on_what_is_covered(smooth_image):
    # The depth map claims a pixel exactly where the colour render of frame 0 put light.
    splat = photo_plane(smooth_image, f_px=80.0, depth=2.0)
    covered = np.asarray(splat.render(size=64, frames=2)[0]).sum(axis=-1) > 0
    assert (depth_map(splat, 64).valid == covered).mean() > 0.99


def test_files_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    z = (0.4 + 40.0 * rng.random((23, 29))).astype(np.float32)
    z[::7, ::5] = np.nan
    dm = DepthMap(depth=z, f_px=123.5, exif_focal=False)

    exact = tmp_path / "z.npz"
    assert write_depth(dm, exact) > 0
    back = load_depth(exact)
    assert np.array_equal(back.depth, dm.depth, equal_nan=True)
    assert (back.f_px, back.exif_focal) == (123.5, False)

    quant = tmp_path / "z.png"
    assert write_depth(dm, quant) > 0
    # The file itself must be 16-bit greyscale (IHDR bit depth 16, colour type 0). Pillow's mode
    # for it varies by version: newer releases open it as "I;16", Pillow 9.1 (our floor) as "I".
    header = quant.read_bytes()[:26]
    assert header[12:16] == b"IHDR" and (header[24], header[25]) == (16, 0)
    with Image.open(quant) as img:
        assert img.mode in ("I;16", "I")
        assert json.loads(img.info["splatgif-depth"])["scale"] == SCALE
    back = load_depth(quant)
    assert (np.isnan(back.depth) == np.isnan(z)).all(), "holes must stay holes"
    assert np.abs(back.depth - z)[np.isfinite(z)].max() <= 0.5 / SCALE + 1e-6
    assert (back.f_px, back.exif_focal) == (123.5, False)


def test_png_saturates_instead_of_wrapping_and_never_mistakes_far_for_empty(tmp_path):
    z = np.array([[0.001, 1.0, 255.0, 300.0, np.nan]], np.float32)
    write_depth(DepthMap(depth=z), tmp_path / "z.png")
    back = load_depth(tmp_path / "z.png").depth[0]
    assert back[0] == pytest.approx(1 / SCALE) and back[1] == pytest.approx(1.0)
    assert back[2] == pytest.approx(255.0, abs=0.01) and back[3] == pytest.approx(65535 / SCALE)
    assert np.isnan(back[4])


def test_pictures_are_for_eyes_and_are_not_loadable(tmp_path):
    z = 1.0 + np.arange(48, dtype=np.float32).reshape(6, 8)
    z[0, 0] = np.nan
    for name in ("z_cmap.png", "z.jpg"):
        path = tmp_path / name
        assert write_depth(DepthMap(depth=z), path, cmap=name.endswith("png")) > 0
        with Image.open(path) as img:
            assert img.mode == "RGB" and img.size == (8, 6)
        with pytest.raises(ValueError, match="not a splatgif depth file"):
            load_depth(path)
    rgb = preview(z)
    assert tuple(rgb[0, 0]) == (128, 128, 128)
    assert tuple(rgb[5, 7]) != tuple(rgb[0, 1]), "near and far must not share a colour"


def test_empty_maps_are_refused(tmp_path):
    with pytest.raises(RuntimeError, match="empty"):
        write_depth(DepthMap(depth=np.full((4, 4), np.nan, np.float32)), tmp_path / "e.png")


def test_cli_depth_mode_end_to_end(fake_model, photo_file, tmp_path, capsys):
    out = tmp_path / "d.png"
    assert cli.main([str(photo_file), "--depth", str(out), "--size", "48", "-q"]) == 0
    assert capsys.readouterr().out.strip() == str(out)
    dm = load_depth(out)
    assert (dm.width, dm.height) == (48, 32)
    assert abs(np.nanmedian(dm.depth) - 3.0) < 0.01  # the fake network's sheet sits at 3
    assert dm.exif_focal is False and dm.f_px == pytest.approx(40.0, rel=0.02)

    bad = tmp_path / "d.jpg"
    assert cli.main([str(photo_file), "--depth", str(bad), "-q"]) == 1
    assert "depth file must end in" in capsys.readouterr().err and not bad.exists()


@pytest.mark.parametrize(
    "argv", [["--depth", "d.png", "-o", "x.gif"], ["--depth-cmap"]], ids=["with -o", "no --depth"]
)
def test_cli_refuses_contradictory_flags(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["photo.jpg", *argv])
    assert exc.value.code == 2


def test_make_depth_reports_the_scene(fake_model, photo_file, tmp_path):
    logs = []
    info = splatgif.make_depth(
        photo_file, tmp_path / "d.npz", size=48, device="cpu", log=logs.append
    )
    assert (info.width, info.height) == (48, 32) and info.bytes > 0 and info.device == "cpu"
    assert 2.99 < info.min_depth <= info.max_depth < 3.01
    assert info.invalid_px == 0 and not info.exif_focal
    assert any("30 mm" in line for line in logs)
    default = splatgif.make_depth(photo_file, size=32, device="cpu")
    assert default.path == photo_file.with_suffix(".depth.png") and default.path.is_file()


def test_splat_depth_method_matches_the_function(fake_model, photo_file):
    s = splatgif.predict(photo_file, device="cpu")
    a, b = s.depth_map(48), depth_map(s, 48)
    assert np.array_equal(a.depth, b.depth, equal_nan=True)
