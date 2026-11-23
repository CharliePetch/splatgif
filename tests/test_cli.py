import subprocess
import sys

import pytest

from splatgif import cli


def parse(*argv):
    return cli.build_parser().parse_args(["photo.jpg", *argv])


def test_defaults():
    a = parse()
    assert (a.size, a.motion, a.frames, a.delay, a.focus) == (640, "moderate", 48, 20, None)


def test_options_parse():
    a = parse(
        "--size",
        "1280",
        "--motion",
        "drastic",
        "--focus",
        "0.5,0.4",
        "--frames",
        "60",
        "--no-dither",
    )
    assert (a.size, a.motion, a.focus, a.frames, a.no_dither) == (
        1280,
        "drastic",
        (0.5, 0.4),
        60,
        True,
    )
    assert parse("--motion", "4.5").motion == 4.5


@pytest.mark.parametrize(
    "bad",
    [["--motion", "wild"], ["--motion", "90"], ["--focus", "0.5"], ["--focus", "2,0.5"],
     ["--size", "4"], ["--frames", "1"], ["--delay", "5"]],
)  # fmt: skip
def test_bad_values_are_refused(bad, capsys):
    with pytest.raises(SystemExit) as exc:
        parse(*bad)
    assert exc.value.code == 2


def test_help_does_not_import_torch():
    code = "import sys, splatgif.cli as c; c.build_parser(); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_photo_is_a_clean_error(tmp_path, capsys):
    assert cli.main([str(tmp_path / "nope.jpg"), "-q"]) == 1
    assert "nope.jpg" in capsys.readouterr().err
