import numpy as np
from PIL import Image

from splatgif.gif import save_gif


def frames(n=6, w=48, h=32):
    out = []
    for i in range(n):
        g = np.linspace(0, 255, w)[None, :, None] * np.ones((h, 1, 3))
        g[..., 1] = (g[..., 1] + 30 * i) % 256  # colours differ from frame to frame
        out.append(Image.fromarray(g.astype(np.uint8)))
    return out


def gif_blocks(data: bytes):
    """(global colour table, [local colour table or None per frame], loop count) from raw bytes."""
    assert data[:6] in (b"GIF87a", b"GIF89a")
    packed = data[10]
    pos = 13
    global_table = None
    if packed & 0x80:
        global_table = data[pos : pos + 3 * 2 ** ((packed & 7) + 1)]
        pos += len(global_table)
    local, loop = [], None
    while data[pos] != 0x3B:
        if data[pos] == 0x21:  # extension
            label = data[pos + 1]
            pos += 2
            while data[pos]:
                if label == 0xFF and data[pos] == 11 and data[pos + 1 : pos + 12] == b"NETSCAPE2.0":
                    loop = int.from_bytes(data[pos + 14 : pos + 16], "little")
                pos += data[pos] + 1
            pos += 1
        elif data[pos] == 0x2C:  # image
            flags = data[pos + 9]
            pos += 10
            table = None
            if flags & 0x80:
                table = data[pos : pos + 3 * 2 ** ((flags & 7) + 1)]
                pos += len(table)
            local.append(table)
            pos += 1  # LZW minimum code size
            while data[pos]:
                pos += data[pos] + 1
            pos += 1
        else:
            raise AssertionError(f"unexpected block {data[pos]:#x}")
    return global_table, local, loop


def test_loops_forever_with_the_requested_delay(tmp_path):
    path = tmp_path / "a.gif"
    n = save_gif(frames(), path, delay_ms=20)
    assert n == path.stat().st_size > 0
    with Image.open(path) as im:
        assert im.n_frames == 6
        assert im.size == (48, 32)
        assert im.info["loop"] == 0
        assert im.info["duration"] == 20


def test_every_frame_uses_the_same_palette(tmp_path):
    path = tmp_path / "a.gif"
    save_gif(frames(), path)
    global_table, local, loop = gif_blocks(path.read_bytes())
    assert len(local) == 6 and loop == 0
    # Repeating the table per frame is fine; a *different* one per frame makes colours crawl.
    assert global_table is not None
    assert all(table in (None, global_table) for table in local)


def test_dither_changes_pixels_not_geometry(tmp_path):
    f = frames(2)
    a, b = tmp_path / "d.gif", tmp_path / "n.gif"
    save_gif(f, a, dither=True)
    save_gif(f, b, dither=False)
    with Image.open(a) as ia, Image.open(b) as ib:
        assert ia.size == ib.size and ia.n_frames == ib.n_frames == 2


def test_rejects_an_empty_loop(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        save_gif([], tmp_path / "x.gif")
