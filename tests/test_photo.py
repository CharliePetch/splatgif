import math
import struct

import numpy as np
import pytest
from PIL import Image

from splatgif.photo import load_photo

DIAG_35 = math.hypot(36, 24)


def exif_block(orientation=None, focal=None, focal_35mm=None):
    """Hand-assembled EXIF, so fixtures do not depend on how a given Pillow writes it."""
    n_exif = (focal is not None) + (focal_35mm is not None)
    n_ifd0 = (orientation is not None) + bool(n_exif)
    exif_at = 8 + 2 + 12 * n_ifd0 + 4
    data_at = exif_at + 2 + 12 * n_exif + 4
    ifd0, exif, data = b"", b"", b""
    if orientation is not None:
        ifd0 += struct.pack(">HHIHH", 0x0112, 3, 1, orientation, 0)  # SHORT, stored inline
    if n_exif:
        ifd0 += struct.pack(">HHII", 0x8769, 4, 1, exif_at)  # pointer to the Exif IFD
    if focal is not None:
        exif += struct.pack(">HHII", 0x920A, 5, 1, data_at)  # RATIONAL: value lives after the IFDs
        data = struct.pack(">II", round(focal * 100), 100)
    if focal_35mm is not None:
        exif += struct.pack(">HHIHH", 0xA405, 3, 1, focal_35mm, 0)
    tiff = struct.pack(">2sHI", b"MM", 42, 8) + struct.pack(">H", n_ifd0) + ifd0 + bytes(4)
    if n_exif:
        tiff += struct.pack(">H", n_exif) + exif + bytes(4) + data
    return b"Exif\x00\x00" + tiff


def jpeg(path, size=(60, 40), **exif):
    Image.fromarray(np.full((size[1], size[0], 3), 128, np.uint8)).save(
        path, exif=exif_block(**exif) if exif else b""
    )
    return path


def test_35mm_equivalent_focal_length(tmp_path):
    photo = load_photo(jpeg(tmp_path / "a.jpg", focal_35mm=50))
    assert photo.exif_focal
    assert photo.f_px == pytest.approx(50 * math.hypot(60, 40) / DIAG_35)


def test_35mm_equivalent_wins_over_the_real_focal_length(tmp_path):
    photo = load_photo(jpeg(tmp_path / "a.jpg", focal=4.2, focal_35mm=26))
    assert photo.f_px == pytest.approx(26 * math.hypot(60, 40) / DIAG_35)


def test_real_focal_length_of_a_phone_sensor_is_scaled(tmp_path):
    photo = load_photo(jpeg(tmp_path / "a.jpg", focal=4.2))
    assert photo.exif_focal
    assert photo.f_px == pytest.approx(4.2 * 8.4 * math.hypot(60, 40) / DIAG_35)


def test_missing_focal_length_assumes_30mm_and_says_so(tmp_path):
    photo = load_photo(jpeg(tmp_path / "a.jpg"))
    assert not photo.exif_focal
    assert photo.f_px == pytest.approx(30 * math.hypot(60, 40) / DIAG_35)


def test_exif_rotation_is_applied_before_sizing(tmp_path):
    photo = load_photo(jpeg(tmp_path / "a.jpg", size=(60, 40), orientation=6))
    assert (photo.width, photo.height) == (40, 60)
    assert photo.rgb.shape == (60, 40, 3)
    assert photo.f_px == pytest.approx(30 * math.hypot(40, 60) / DIAG_35)


def test_png_and_greyscale_load_as_rgb(tmp_path):
    Image.fromarray(np.full((20, 30), 90, np.uint8)).save(tmp_path / "g.png")
    photo = load_photo(tmp_path / "g.png")
    assert photo.rgb.shape == (20, 30, 3) and photo.rgb.dtype == np.uint8
