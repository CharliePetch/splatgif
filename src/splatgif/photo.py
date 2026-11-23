"""Reading a photo: upright RGB pixels plus the focal length SHARP needs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_FOCAL_LENGTH = 0x920A  # EXIF FocalLength, mm
_FOCAL_LENGTH_35MM = 0xA405  # EXIF FocalLengthIn35mmFilm, mm
_EXIF_IFD = 0x8769
_FALLBACK_35MM = 30.0  # what Apple's tooling assumes when a photo carries no focal length


@dataclass(frozen=True)
class Photo:
    rgb: np.ndarray  # (H, W, 3) uint8, EXIF orientation applied
    f_px: float  # focal length in pixels of this image
    exif_focal: bool  # False: the file had no usable focal length and 30 mm was assumed

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]


def _focal_35mm(tags: dict) -> tuple[float, bool]:
    """35 mm-equivalent focal length from EXIF, mirroring Apple's ``load_rgb``."""

    def positive(tag: int) -> float | None:
        try:
            value = float(tags[tag])
        except (KeyError, TypeError, ValueError):
            return None
        return value if value > 0 else None

    f35 = positive(_FOCAL_LENGTH_35MM)
    if f35 is not None and f35 >= 1:
        return f35, True
    f = positive(_FOCAL_LENGTH)
    if f is None:
        return _FALLBACK_35MM, False
    # A real focal length below 10 mm is a phone/compact sensor, not 35 mm film: crude conversion.
    return (f * 8.4 if f < 10 else f), True


def load_photo(path: str | Path) -> Photo:
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            import pillow_heif
        except ImportError as exc:
            raise ImportError("HEIC photos need: pip install 'splatgif[heic]'") from exc
        pillow_heif.register_heif_opener()

    with Image.open(path) as img:
        exif = img.getexif()
        tags = {**exif, **exif.get_ifd(_EXIF_IFD)}
        rgb = np.array(ImageOps.exif_transpose(img).convert("RGB"))

    f_35mm, from_exif = _focal_35mm(tags)
    height, width = rgb.shape[:2]
    f_px = f_35mm * math.hypot(width, height) / math.hypot(36, 24)
    return Photo(rgb=rgb, f_px=f_px, exif_focal=from_exif)
