"""Command line: ``splatgif photo.jpg``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .motion import MOTIONS


def _motion(text: str) -> str | float:
    if text in MOTIONS:
        return text
    try:
        degrees = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"use {' / '.join(MOTIONS)} or a number of degrees"
        ) from None
    if not 0 < degrees <= 45:
        raise argparse.ArgumentTypeError("degrees must be in (0, 45]")
    return degrees


def _focus(text: str) -> tuple[float, float]:
    try:
        x, y = (float(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected X,Y as fractions, e.g. 0.5,0.4") from None
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise argparse.ArgumentTypeError("X and Y must each be between 0 and 1")
    return x, y


def _positive(minimum: int):
    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="splatgif",
        description="Turn one photo into a looping 3D 'wigglegram' GIF using Apple's SHARP.",
    )
    p.add_argument("image", help="JPEG, PNG, WebP, TIFF or (with splatgif[heic]) HEIC")
    p.add_argument("-o", "--output", help="output path (default: next to the photo, .gif)")
    p.add_argument("--size", type=_positive(16), default=640, help="long edge in px (default 640)")
    p.add_argument(
        "--motion",
        type=_motion,
        default="moderate",
        metavar="MODE",
        help="moderate (+-2.5 deg, default), drastic (+-8 deg) or the peak yaw in degrees",
    )
    p.add_argument("--frames", type=_positive(2), default=48, help="frames per loop (default 48)")
    p.add_argument(
        "--delay",
        type=_positive(10),
        default=20,
        metavar="MS",
        help="per-frame delay in ms (default 20, the fastest browsers honour)",
    )
    p.add_argument(
        "--focus",
        type=_focus,
        metavar="X,Y",
        help="pixel to pivot on, as fractions of width,height (default: the centre)",
    )
    p.add_argument("--no-dither", action="store_true", help="smaller file, more banding")
    p.add_argument("--device", help="cuda, mps or cpu (default: the best available)")
    p.add_argument("--weights", metavar="PATH", help="SHARP checkpoint (default: download once)")
    p.add_argument("--accept-license", action="store_true",
                   help="accept Apple's research-only model licence without prompting")  # fmt: skip
    p.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    p.add_argument("--version", action="version", version=f"splatgif {__version__}")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from .pipeline import make_gif

    try:
        info = make_gif(
            args.image,
            args.output,
            size=args.size,
            motion=args.motion,
            frames=args.frames,
            delay_ms=args.delay,
            focus=args.focus,
            dither=not args.no_dither,
            device=args.device,
            weights=args.weights,
            accept_license=args.accept_license,
            log=None if args.quiet else lambda msg: print(f"splatgif: {msg}", file=sys.stderr),
        )
    except (FileNotFoundError, RuntimeError, ImportError, OSError) as exc:
        print(f"splatgif: {exc}", file=sys.stderr)
        return 1
    print(info.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
