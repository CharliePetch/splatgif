"""Finding (and, with consent, downloading) Apple's SHARP checkpoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
SHA256 = "94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d"
LICENSE_URL = "https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL"
ENV_PATH = "SPLATGIF_WEIGHTS"
ENV_ACCEPT = "SPLATGIF_ACCEPT_LICENSE"


def default_path() -> Path:
    """Where ``torch.hub`` caches it, i.e. also where Apple's own ``sharp`` CLI puts it."""
    return Path(torch.hub.get_dir()) / "checkpoints" / URL.rsplit("/", 1)[-1]


def _require_consent(accepted: bool) -> None:
    if accepted or os.environ.get(ENV_ACCEPT) == "1":
        return
    notice = (
        "splatgif needs Apple's SHARP weights (2.8 GB). They are licensed for non-commercial\n"
        f"research use only, under revocable terms: {LICENSE_URL}\n"
    )
    if sys.stdin.isatty() and sys.stderr.isatty():
        print(notice, file=sys.stderr)
        if input("Download them and accept that licence? [y/N] ").strip().lower() in {"y", "yes"}:
            return
        raise SystemExit("Declined; nothing downloaded.")
    raise RuntimeError(
        notice + f"Pass accept_license=True / --accept-license (or set {ENV_ACCEPT}=1) to download."
    )


def resolve_weights(path: str | Path | None = None, accept_license: bool = False) -> Path:
    """An explicit path (or $SPLATGIF_WEIGHTS), else the cached download, else download it."""
    explicit = path or os.environ.get(ENV_PATH)
    if explicit:
        explicit = Path(explicit).expanduser()
        if not explicit.is_file():
            raise FileNotFoundError(f"SHARP weights not found: {explicit}")
        return explicit

    cached = default_path()
    if cached.is_file():
        return cached

    _require_consent(accept_license)
    cached.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SHARP weights to {cached}", file=sys.stderr)
    torch.hub.download_url_to_file(URL, str(cached), hash_prefix=SHA256, progress=True)
    return cached
