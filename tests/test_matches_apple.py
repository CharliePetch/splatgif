"""The condensed network against Apple's own ml-sharp, on the same photo.

Needs Apple's package (``pip install git+https://github.com/apple/ml-sharp``, deliberately not a
dependency of splatgif), the cached weights and about 10 GB of RAM, so it only runs under
``pytest -m slow`` in an environment that has all three. Re-run it when ml-sharp or the checkpoint
changes.
"""

import gc

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from splatgif import model, weights
from splatgif.photo import Photo

pytestmark = pytest.mark.slow

pytest.importorskip("sharp")
if not weights.default_path().is_file():
    pytest.skip("SHARP weights are not cached", allow_module_level=True)

INTERNAL = (1536, 1536)


def photo(w=640, h=480) -> Photo:
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([80 + 120 * x / w, 60 + 140 * y / h, 150 - 60 * x / w], axis=-1)
    img[(x - w * 0.4) ** 2 + (y - h * 0.5) ** 2 < 90**2] = [210, 70, 60]
    img += rng.normal(0, 4, img.shape)
    return Photo(
        np.clip(img, 0, 255).astype(np.uint8),
        f_px=30 * np.hypot(w, h) / np.hypot(36, 24),
        exif_focal=False,
    )


def apple_gaussians(p: Photo, checkpoint):
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils.gaussians import unproject_gaussians

    net = create_predictor(PredictorParams())
    net.load_state_dict(torch.load(checkpoint, weights_only=True))
    net.eval()
    with torch.no_grad():
        x = torch.from_numpy(p.rgb.copy()).float().permute(2, 0, 1)[None] / 255.0
        x = F.interpolate(x, size=INTERNAL, mode="bilinear", align_corners=True)
        ndc = net(x, torch.tensor([p.f_px / p.width]).float())
        k = torch.tensor(
            [[p.f_px, 0, p.width / 2, 0], [0, p.f_px, p.height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        k = k.float()
        k[0] *= INTERNAL[0] / p.width
        k[1] *= INTERNAL[1] / p.height
        g = unproject_gaussians(ndc, torch.eye(4), k, INTERNAL)
    return {name: getattr(g, name)[0].numpy() for name in g._fields}


def test_gaussians_match_apples(tmp_path):
    from sharp.utils.color_space import linearRGB2sRGB

    p = photo()
    checkpoint = weights.default_path()
    ref = apple_gaussians(p, checkpoint)
    gc.collect()

    net = model.load_predictor(checkpoint, "cpu")
    with torch.inference_mode():
        x = torch.from_numpy(p.rgb.copy()).float().permute(2, 0, 1)[None] / 255.0
        x = F.interpolate(x, size=INTERNAL, mode="bilinear", align_corners=True)
        ours = model.to_camera_space(net(x, p.f_px, p.width), p.f_px, p.width, p.height)

    # Apple stores each Gaussian as quaternion + scales; rebuild the covariance to compare.
    r = model._rotation_matrices(torch.from_numpy(ref["quaternions"])).numpy().astype(np.float64)
    s = ref["singular_values"].astype(np.float64)
    cov = np.einsum("nij,nj,nkj->nik", r, s * s, r)
    cov6 = cov.reshape(-1, 9)[:, [0, 1, 2, 4, 5, 8]]
    trace = (cov6[:, 0] + cov6[:, 3] + cov6[:, 5])[:, None]

    means_rel = np.abs(ours.means.numpy() - ref["mean_vectors"]) / np.abs(ref["mean_vectors"]).max(
        1, keepdims=True
    )
    cov_rel = np.abs(ours.cov.numpy() - cov6) / trace
    colors = np.abs(ours.colors.numpy() - linearRGB2sRGB(torch.from_numpy(ref["colors"])).numpy())
    opacity = np.abs(ours.opacities.numpy() - ref["opacities"])

    print(  # visible with `pytest -s`
        f"\nvs Apple, {len(opacity):,} Gaussians:"
        f"\n  means: p99.9 {np.percentile(means_rel, 99.9):.1e}, max {means_rel.max():.1e}"
        f"\n  covariance / trace: p99.9 {np.percentile(cov_rel, 99.9):.1e}, max {cov_rel.max():.1e}"
        f"\n  colour max {colors.max():.1e}, opacity max {opacity.max():.1e}"
    )
    assert np.percentile(means_rel, 99.9) < 1e-4
    assert np.percentile(cov_rel, 99.9) < 1e-3
    assert colors.max() < 1e-3
    assert opacity.max() < 2e-3
