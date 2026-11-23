"""The predicted scene (a Gaussian splat seen from the photo's own camera) and how to render it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import INTERNAL, load_predictor, to_camera_space
from .motion import frame_size, sway
from .photo import Photo, load_photo
from .weights import resolve_weights

_FOCUS_WINDOW = 0.1  # fraction of the frame, per axis, that counts as "the subject"
_FOCUS_MIN_SAMPLES = 500


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Splat:
    """Gaussians in the photo camera's frame (x right, y down, z forward, camera at the origin)."""

    means: np.ndarray  # (N, 3) float32
    cov: np.ndarray  # (N, 6) float32: xx xy xz yy yz zz
    colors: np.ndarray  # (N, 3) float32, sRGB in [0, 1]
    opacity: np.ndarray  # (N,) float32
    f_px: float  # the photo's focal length in pixels
    width: int  # the photo's size
    height: int
    exif_focal: bool = True  # False: the photo had no focal length, depth is only known up to scale

    def focus_point(self, focus: tuple[float, float] | None = None) -> np.ndarray:
        """The 3D point the sway pivots on.

        ``focus`` is a pixel as (x, y) fractions of the frame, default the centre. The point lies
        on the ray through that pixel at the median depth of the Gaussians that project near it.
        """
        x, y, z = self.means.T
        front = z > 0
        # Offsets of the chosen pixel and of every Gaussian from the principal point, in pixels.
        du = 0.0 if focus is None else focus[0] * self.width - self.width / 2
        dv = 0.0 if focus is None else focus[1] * self.height - self.height / 2
        u, v = self.f_px * x[front] / z[front], self.f_px * y[front] / z[front]
        near = (np.abs(u - du) <= self.width * _FOCUS_WINDOW / 2) & (
            np.abs(v - dv) <= self.height * _FOCUS_WINDOW / 2
        )
        if near.sum() < _FOCUS_MIN_SAMPLES:  # too sparse to be the subject: use the whole scene
            return np.array([0.0, 0.0, np.median(z[front])])
        depth = float(np.median(z[front][near]))
        return np.array([du * depth / self.f_px, dv * depth / self.f_px, depth])

    def render(
        self,
        size: int = 640,
        motion: str | float = "moderate",
        frames: int = 48,
        focus: tuple[float, float] | None = None,
    ) -> list[Image.Image]:
        """Frames of a looping side-to-side sway about the focus point.

        Frame 0 is the photograph itself. ``size`` is the long edge in pixels, ``motion`` a preset
        name or the peak yaw in degrees, ``frames`` the number of frames in one loop.
        """
        from .raster import render_frame  # numba: imported late so `--help` stays instant

        w, h = frame_size(size, self.width / self.height)
        fx, fy = self.f_px * w / self.width, self.f_px * h / self.height
        pivot = self.focus_point(focus)
        return [
            Image.fromarray(render_frame(self, yaw, pivot, w, h, fx, fy))
            for yaw in sway(motion, frames)
        ]


def predict(
    photo: Photo | str | Path,
    *,
    weights: str | Path | None = None,
    device: str | None = None,
    accept_license: bool = False,
) -> Splat:
    """Run SHARP on one photo. The slow step (about two minutes on a CPU, seconds on a GPU)."""
    if not isinstance(photo, Photo):
        photo = load_photo(photo)
    dev = pick_device(device)
    model = load_predictor(resolve_weights(weights, accept_license), dev)

    with torch.inference_mode():
        image = torch.from_numpy(photo.rgb).to(dev).permute(2, 0, 1)[None].float() / 255
        image = F.interpolate(image, size=(INTERNAL, INTERNAL), mode="bilinear", align_corners=True)
        raw = model(image, photo.f_px, photo.width)
        del image, model
        g = to_camera_space(raw, photo.f_px, photo.width, photo.height)
        keep = g.opacities >= 1 / 255  # anything fainter can never reach one 8-bit level
        arrays = [t[keep].cpu().numpy() for t in g]
    means, cov, colors, opacity = arrays
    return Splat(
        means, cov, colors, opacity, photo.f_px, photo.width, photo.height, photo.exif_focal
    )
