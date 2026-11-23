"""The whole photo -> GIF path with a stand-in for the SHARP network (no weights needed)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

import splatgif
from splatgif import model, splat


class FakePredictor:
    """Describes the photo as a flat sheet of Gaussians, in the network's own NDC terms."""

    def __call__(self, image, f_px, width):
        g, z = 48, 3.0
        colors = F.interpolate(image, size=(g, g), mode="area")[0]
        ndc = 2 * (torch.arange(g) + 0.5) / g - 1
        gy, gx = torch.meshgrid(ndc, ndc, indexing="ij")
        n = g * g
        return model.Raw(
            means=torch.stack(
                [z * gx.reshape(-1), z * gy.reshape(-1), torch.full((n,), z)], dim=-1
            ),
            scales=torch.full((n, 3), 0.6 * z * 2 / g),
            quats=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
            colors=colors.reshape(3, -1).T.contiguous(),
            opacities=torch.full((n,), 0.9),
        )


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(splat, "resolve_weights", lambda *a, **k: "unused.pt")
    monkeypatch.setattr(splat, "load_predictor", lambda *a, **k: FakePredictor())


@pytest.fixture
def photo_file(tmp_path, smooth_image):
    path = tmp_path / "scene.jpg"
    # 96 x 64: landscape, so the x and y halves of the NDC -> camera map differ
    Image.fromarray(smooth_image).resize((96, 64), Image.Resampling.BILINEAR).save(path, quality=98)
    return path


def test_frame_zero_reproduces_the_photo_through_the_whole_pipeline(fake_model, photo_file):
    s = splatgif.predict(photo_file, device="cpu")
    assert s.means.dtype == np.float32 and s.cov.shape == (len(s.means), 6)
    assert (s.width, s.height) == (96, 64) and not s.exif_focal
    frame0 = np.asarray(s.render(size=96, frames=4)[0]).astype(int)
    photo = np.asarray(Image.open(photo_file).convert("RGB")).astype(int)
    assert frame0.shape == photo.shape
    assert (
        np.abs(frame0 - photo).mean() < 4.0
    )  # a mirrored or half-pixel-off sheet would be far worse


def test_make_gif_writes_a_looping_gif_and_reports_it(fake_model, photo_file, tmp_path):
    logs = []
    out = tmp_path / "out.gif"
    info = splatgif.make_gif(photo_file, out, size=96, frames=6, device="cpu", log=logs.append)
    assert info.path == out and out.stat().st_size == info.bytes > 0
    assert (info.width, info.height, info.frames, info.delay_ms) == (96, 64, 6, 20)
    assert not info.exif_focal and info.device == "cpu"
    assert info.focus_depth == pytest.approx(3.0, rel=1e-3)
    assert any("30 mm" in line for line in logs)
    with Image.open(out) as gif:
        assert gif.size == (96, 64) and gif.info["loop"] == 0
        # Pillow folds identical neighbouring frames into one longer frame, so the count can drop
        # (this flat test scene barely changes) but the loop length may not.
        total = 0
        for i in range(gif.n_frames):
            gif.seek(i)
            total += gif.info["duration"]
        assert total == 6 * 20


def test_default_output_sits_next_to_the_photo(fake_model, photo_file):
    info = splatgif.make_gif(photo_file, size=64, frames=3, device="cpu")
    assert info.path == photo_file.with_suffix(".gif") and info.path.exists()


def test_cli_prints_the_output_path(fake_model, photo_file, tmp_path, capsys):
    from splatgif import cli

    out = tmp_path / "cli.gif"
    assert (
        cli.main(
            [
                str(photo_file),
                "-o",
                str(out),
                "--size",
                "64",
                "--frames",
                "3",
                "--device",
                "cpu",
                "-q",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(out)
    assert out.exists()
