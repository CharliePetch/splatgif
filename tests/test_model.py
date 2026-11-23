import gzip
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from splatgif import model


def test_architecture_matches_the_released_checkpoint():
    """Parameter names and shapes recorded from Apple's checkpoint (unused tensors dropped)."""
    expected = (
        gzip.open(Path(__file__).parent / "sharp_params.txt.gz", "rt").read().split("\n")[:-1]
    )
    with torch.device("meta"):
        net = model.Predictor()
    ours = sorted(f"{k} {'x'.join(map(str, v.shape))}" for k, v in net.state_dict().items())
    assert ours == expected


@pytest.mark.parametrize(
    ("size", "overlap", "pad", "side"), [(1536, 0.25, 3, 96), (768, 0.5, 6, 48)]
)
def test_tiles_stitch_back_exactly(size, overlap, pad, side):
    """Pooling each overlapping tile to its feature grid and merging recovers the pooled image."""
    x = torch.randn(1, 2, size, size)
    tiles = model._split(x, overlap)
    assert len(tiles) == (5 if size == 1536 else 3) ** 2
    features = F.avg_pool2d(tiles, 16)  # 384 px tile -> 24 x 24 feature cells
    merged = model._merge(features, pad)
    assert merged.shape == (1, 2, side, side)
    torch.testing.assert_close(merged, F.avg_pool2d(x, 16))


def test_rotation_matrices():
    q = torch.tensor([[1.0, 0, 0, 0], [math.sqrt(0.5), 0, 0, math.sqrt(0.5)], [2.0, 0, 0, 0]])
    r = model._rotation_matrices(q)
    torch.testing.assert_close(r[0], torch.eye(3))
    torch.testing.assert_close(r[2], torch.eye(3))  # not required to be unit length
    # 90 degrees about +z takes +x to +y
    torch.testing.assert_close(
        r[1] @ torch.tensor([1.0, 0, 0]), torch.tensor([0.0, 1.0, 0]), atol=1e-6, rtol=0
    )
    rng = torch.randn(50, 4)
    rr = model._rotation_matrices(rng)
    torch.testing.assert_close(
        rr @ rr.transpose(1, 2), torch.eye(3).expand(50, 3, 3), atol=1e-5, rtol=0
    )
    assert torch.allclose(torch.linalg.det(rr), torch.ones(50), atol=1e-5)


def test_camera_space_is_a_diagonal_map_of_the_covariance():
    f, w, h = 500.0, 400, 300
    n = 20
    g = torch.Generator().manual_seed(0)
    raw = model.Raw(
        means=torch.randn(n, 3, generator=g),
        scales=torch.rand(n, 3, generator=g) + 0.1,
        quats=torch.randn(n, 4, generator=g),
        colors=torch.rand(n, 3, generator=g),
        opacities=torch.rand(n, generator=g),
    )
    out = model.to_camera_space(raw, f, w, h)
    d = np.diag([w / (2 * f), h / (2 * f), 1.0])
    for i in range(n):
        r = model._rotation_matrices(raw.quats[i : i + 1])[0].numpy().astype(np.float64)
        cov = d @ r @ np.diag(raw.scales[i].numpy().astype(np.float64) ** 2) @ r.T @ d.T
        xx, xy, xz, yy, yz, zz = out.cov[i].numpy()
        np.testing.assert_allclose([xx, xy, xz, yy, yz, zz], cov[np.triu_indices(3)], rtol=1e-4)
        np.testing.assert_allclose(out.means[i].numpy(), d @ raw.means[i].numpy(), rtol=1e-5)


def test_vit_matches_timm():
    """The hand-written ViT is a drop-in for timm's (skipped when timm is not installed)."""
    timm = pytest.importorskip("timm")
    kw = dict(
        embed_dim=64,
        depth=3,
        num_heads=4,
        img_size=32,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
    )
    ref = timm.models.VisionTransformer(global_pool="", **kw).eval()
    ours = model._ViT(taps=(0, 1), dim=64, depth=3, heads=4, img=32, patch=16).eval()
    state = {k: v for k, v in ref.state_dict().items() if not k.startswith(("head", "fc_norm"))}
    for v in state.values():  # make every parameter non-trivial
        v.copy_(torch.randn_like(v) * 0.2)
    ours.load_state_dict(state, strict=True)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        want = ref.forward_features(x)[:, 1:]  # (B, tokens, C), class token dropped
        got, taps = ours(x)
    torch.testing.assert_close(got.flatten(2).transpose(1, 2), want, atol=1e-5, rtol=1e-4)
    assert len(taps) == 2 and taps[0].shape == (2, 64, 2, 2)
