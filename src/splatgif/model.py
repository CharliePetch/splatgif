"""The SHARP network, condensed to what inference with the released checkpoint needs.

Derived from Apple's ml-sharp (https://github.com/apple/ml-sharp, Copyright (C) 2025 Apple Inc.;
see ``licenses/APPLE-ML-SHARP-LICENSE``). This is a modified, much smaller re-implementation:

* only the released configuration is kept (DINOv2-L/16 @ 384 px, 1536 px input, two Gaussian
  layers at stride 2); every training / alternative-config branch is gone;
* the ViT is written out here instead of importing ``timm`` (and, through it, torchvision);
* the depth-alignment UNet and the ViT classifier heads, which inference never touches, are
  not instantiated (their weights are dropped when the checkpoint is loaded);
* nothing is converted to or from quaternions: Gaussians leave this module as camera-space
  covariance matrices, which is what the rasteriser consumes;
* colours are left as the sRGB values the network produces. Apple converts them to linear RGB (to
  blend there) and, for PLY export, back again; the rasteriser here blends sRGB directly.

Module and parameter names are kept identical to the checkpoint so it loads with ``strict=True``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

INTERNAL = 1536  # the network always sees the photo resampled to INTERNAL x INTERNAL
STRIDE = 2  # Gaussians sit on an (INTERNAL / STRIDE)^2 grid ...
LAYERS = 2  # ... two layers deep: 2 * 768 * 768 = 1,179,648 Gaussians

# Checkpoint tensors that inference never reads.
_UNUSED = re.compile(r"^depth_alignment\.|\.(patch|image)_encoder\.head\.")


# --------------------------------------------------------------------------------------
# ViT-L/16 @ 384 (the DINOv2 variant Apple used inside Depth Pro's sliding pyramid)
# --------------------------------------------------------------------------------------
class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class _LayerScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class _Mlp(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim)
        self.ls2 = _LayerScale(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class _PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)


class _ViT(nn.Module):
    """Returns the final feature grid plus the grids after the blocks listed in ``taps``."""

    def __init__(self, taps: tuple[int, ...] = (), dim=1024, depth=24, heads=16, img=384, patch=16):
        super().__init__()
        self.taps = taps
        self.grid = img // patch
        self.cls_token = nn.Parameter(torch.empty(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.empty(1, self.grid**2 + 1, dim))
        self.patch_embed = _PatchEmbed(dim, patch)
        self.blocks = nn.ModuleList(_Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def _grid(
        self, tokens: Tensor
    ) -> Tensor:  # (B, 1 + g*g, C) -> (B, C, g, g), class token dropped
        b, _, c = tokens.shape
        return tokens[:, 1:].reshape(b, self.grid, self.grid, c).permute(0, 3, 1, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], dim=1) + self.pos_embed
        taps = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.taps:
                taps.append(self._grid(x))
        return self._grid(self.norm(x)), taps


# --------------------------------------------------------------------------------------
# Sliding pyramid encoder (Depth Pro): 25 + 9 + 1 overlapping 384 px tiles through one ViT
# --------------------------------------------------------------------------------------
def _split(x: Tensor, overlap: float, tile: int = 384) -> Tensor:
    stride = int(tile * (1 - overlap))
    steps = math.ceil((x.shape[-1] - tile) / stride) + 1
    return torch.cat(
        [
            x[..., j * stride : j * stride + tile, i * stride : i * stride + tile]
            for j in range(steps)
            for i in range(steps)
        ]
    )


def _merge(tiles: Tensor, pad: int) -> Tensor:
    """Stitch ``steps * steps`` feature tiles back together, trimming ``pad`` cells of overlap."""
    steps = math.isqrt(len(tiles))
    rows = []
    for j in range(steps):
        row = []
        for i in range(steps):
            t = tiles[j * steps + i : j * steps + i + 1]
            if j:
                t = t[..., pad:, :]
            if i:
                t = t[..., :, pad:]
            if j != steps - 1:
                t = t[..., : t.shape[-2] - pad, :]
            if i != steps - 1:
                t = t[..., :, : t.shape[-1] - pad]
            row.append(t)
        rows.append(torch.cat(row, dim=-1))
    return torch.cat(rows, dim=-2)


def _project_upsample(
    dim_in: int, dim_out: int, ups: int, dim_mid: int | None = None
) -> nn.Sequential:
    mid = dim_mid or dim_out
    return nn.Sequential(
        nn.Conv2d(dim_in, mid, 1, bias=False),
        *[
            nn.ConvTranspose2d(mid if i == 0 else dim_out, dim_out, 2, 2, bias=False)
            for i in range(ups)
        ],
    )


_ENC_DIMS = (256, 256, 512, 1024, 1024)


class _SlidingPyramid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        e = _ENC_DIMS
        # Only blocks 5 and 11 of the patch ViT feed the decoder (Apple also hooks 17 and 23,
        # but never reads them).
        self.patch_encoder = _ViT(taps=(5, 11))
        self.image_encoder = _ViT()
        self.upsample_latent0 = _project_upsample(1024, e[0], 3, dim_mid=e[1])
        self.upsample_latent1 = _project_upsample(1024, e[1], 2)
        self.upsample0 = _project_upsample(1024, e[2], 1)
        self.upsample1 = _project_upsample(1024, e[3], 1)
        self.upsample2 = _project_upsample(1024, e[4], 1)
        self.upsample_lowres = nn.ConvTranspose2d(1024, e[4], 2, 2)
        self.fuse_lowres = nn.Conv2d(e[4] * 2, e[4], 1)

    def forward(self, x: Tensor) -> list[Tensor]:  # x: (1, 3, 1536, 1536), range [-1, 1]
        x1 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        p0, p1 = _split(x, 0.25), _split(x1, 0.5)  # 5x5 and 3x3 tiles
        n0, n1 = len(p0), len(p1)

        encoded, (tap5, tap11) = self.patch_encoder(torch.cat([p0, p1, x2]))
        latent0 = _merge(tap5[:n0], 3)
        latent1 = _merge(tap11[:n0], 3)
        e0, e1, e2 = encoded.split([n0, n1, 1])
        feat0, feat1 = _merge(e0, 3), _merge(e1, 6)
        del encoded, tap5, tap11, p0, p1

        lowres, _ = self.image_encoder(x2)
        lowres = self.fuse_lowres(torch.cat([self.upsample2(e2), self.upsample_lowres(lowres)], 1))
        return [
            self.upsample_latent0(latent0),
            self.upsample_latent1(latent1),
            self.upsample0(feat0),
            self.upsample1(feat1),
            lowres,
        ]


# --------------------------------------------------------------------------------------
# DPT-style decoders
# --------------------------------------------------------------------------------------
class _Residual(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),  # safe: acts on a fresh conv output
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.residual(x)


class _Fusion(nn.Module):
    def __init__(self, dim: int, upsample: bool) -> None:
        super().__init__()
        self.resnet1 = _Residual(dim)
        self.resnet2 = _Residual(dim)
        self.deconv = nn.ConvTranspose2d(dim, dim, 2, 2, bias=False) if upsample else nn.Identity()
        self.out_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        if skip is not None:
            x = x + self.resnet1(skip)
        return self.out_conv(self.deconv(self.resnet2(x)))


class _Decoder(nn.Module):
    """Fuses a five-level feature pyramid from the coarsest level up to the finest."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        e = _ENC_DIMS
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(e[0], dim, 1, bias=False) if e[0] != dim else nn.Identity(),
                *[nn.Conv2d(d, dim, 3, padding=1, bias=False) for d in e[1:]],
            ]
        )
        self.fusions = nn.ModuleList(_Fusion(dim, upsample=i > 0) for i in range(len(e)))

    def forward(self, feats: list[Tensor]) -> Tensor:
        x = self.fusions[-1](self.convs[-1](feats[-1]))
        for i in range(len(feats) - 2, -1, -1):
            x = self.fusions[i](x, self.convs[i](feats[i]))
        return x


class _Dpt(nn.Module):
    """Depth Pro: sliding pyramid -> decoder -> two-channel disparity at full resolution."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _SlidingPyramid()
        self.decoder = _Decoder(256)
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ConvTranspose2d(128, 128, 2, 2),
            nn.Conv2d(128, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, LAYERS, 1),
            nn.ReLU(),
        )


class _Monodepth(nn.Module):
    """Wrapper kept only so parameter names match the checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.monodepth_predictor = _Dpt()

    def forward(self, image: Tensor) -> tuple[Tensor, list[Tensor]]:
        dpt = self.monodepth_predictor
        features = dpt.encoder(image * 2 - 1)
        disparity = dpt.head(dpt.decoder(features))
        return disparity, features


class _GroupedResidual(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = dim // 2
        self.residual = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.ReLU(),
            nn.Conv2d(dim, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, dim, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.residual(x)


def _head(dim: int, out: int) -> nn.Sequential:
    return nn.Sequential(
        _GroupedResidual(dim), _GroupedResidual(dim), nn.ReLU(), nn.Conv2d(dim, out, 1), nn.ReLU()
    )


class _SkipConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(5, 128, STRIDE, stride=STRIDE)  # RGB + two disparity layers


class _GaussianDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _Decoder(128)
        self.image_encoder = _SkipConv()
        self.fusion = _Fusion(128, upsample=False)
        self.texture_head = _head(128, 32)
        self.geometry_head = _head(128, 32)

    def forward(self, features_in: Tensor, encodings: list[Tensor]) -> tuple[Tensor, Tensor]:
        x = self.fusion(self.decoder(encodings), self.image_encoder.conv(features_in))
        return self.texture_head(x), self.geometry_head(x)


class _PredictionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.geometry_prediction_head = nn.Conv2d(32, 3 * LAYERS, 1)
        self.texture_prediction_head = nn.Conv2d(32, 11 * LAYERS, 1)

    def forward(self, texture: Tensor, geometry: Tensor) -> Tensor:  # -> (1, 14, LAYERS, H, W)
        g = self.geometry_prediction_head(geometry).unflatten(1, (3, LAYERS))
        t = self.texture_prediction_head(texture).unflatten(1, (11, LAYERS))
        return torch.cat([g, t], dim=1)


# --------------------------------------------------------------------------------------
# Base Gaussians, deltas -> Gaussians
# --------------------------------------------------------------------------------------
def _inverse_sigmoid(x: Tensor) -> Tensor:
    return torch.log(x / (1.0 - x))


def _inverse_softplus(x: Tensor, eps: float = 1e-6) -> Tensor:
    x = x.clamp_min(eps)
    s = torch.sigmoid(-x)
    return x + torch.log(-(s / (1.0 - s)) + 1.0)


class Raw(NamedTuple):
    """What the network predicts, in its own normalised-device parametrisation."""

    means: Tensor  # (N, 3)
    scales: Tensor  # (N, 3) per-axis standard deviations
    quats: Tensor  # (N, 4) wxyz, not normalised
    colors: Tensor  # (N, 3) sRGB in [0, 1]
    opacities: Tensor  # (N,)


class Gaussians(NamedTuple):
    """Gaussians in the photo camera's frame: x right, y down, z forward, camera at the origin."""

    means: Tensor  # (N, 3)
    cov: Tensor  # (N, 6): xx xy xz yy yz zz
    colors: Tensor  # (N, 3) sRGB in [0, 1]
    opacities: Tensor  # (N,)


def _rotation_matrices(q: Tensor) -> Tensor:
    """wxyz quaternions (N, 4), normalised here -> rotation matrices (N, 3, 3)."""
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)  # fmt: skip


def to_camera_space(raw: Raw, f_px: float, width: int, height: int) -> Gaussians:
    """Undo the NDC parametrisation.

    With the principal point at the image centre, unprojection is the diagonal map
    ``diag(W / 2f, H / 2f, 1)``. Apple applies it to quaternion + scale, then re-derives a
    quaternion with an SVD; applying it to the covariance directly gives the same Gaussians.
    """
    d = torch.tensor([width / (2 * f_px), height / (2 * f_px), 1.0], device=raw.means.device)
    m = _rotation_matrices(raw.quats) * raw.scales[:, None, :]  # R diag(s)
    m = m * d[None, :, None]  # D R diag(s)
    c = m @ m.transpose(1, 2)
    cov = torch.stack(
        [c[:, 0, 0], c[:, 0, 1], c[:, 0, 2], c[:, 1, 1], c[:, 1, 2], c[:, 2, 2]], dim=-1
    )
    return Gaussians(raw.means * d, cov, raw.colors, raw.opacities)


class Predictor(nn.Module):
    """Photo (1, 3, 1536, 1536) in [0, 1] -> ``Raw`` Gaussians. Weights: ``load_predictor``."""

    def __init__(self) -> None:
        super().__init__()
        self.monodepth_model = _Monodepth()
        self.feature_model = _GaussianDecoder()
        self.prediction_head = _PredictionHead()

    @torch.inference_mode()
    def forward(self, image: Tensor, f_px: float, width: int) -> Raw:
        """``f_px`` and ``width`` describe the *original* photo, not the resampled input."""
        disparity, encodings = self.monodepth_model(image)
        depth = (f_px / width) / disparity.clamp(1e-4, 1e4)  # (1, 2, 1536, 1536), two depth layers
        del disparity

        # Rescale so the nearest point sits at depth 1; undone through `global_scale` below.
        factor = 1.0 / (depth.amin() + 1e-6)
        depth = (depth * factor).clamp(max=1e2)
        global_scale = 1.0 / factor

        disp = F.max_pool2d(1.0 / depth, STRIDE, STRIDE)[
            :, None
        ]  # (1, 1, 2, h, w): nearest per block
        features_in = 2.0 * torch.cat([image, 1.0 / depth], dim=1) - 1.0
        base_colors = F.avg_pool2d(image, STRIDE, STRIDE)[:, :, None].repeat(1, 1, LAYERS, 1, 1)
        del depth

        texture, geometry = self.feature_model(features_in, encodings)
        del features_in, encodings
        delta = self.prediction_head(texture, geometry)  # (1, 14, 2, h, w)
        del texture, geometry

        h, w = disp.shape[-2:]
        ndc = 2 * torch.arange(0.5 * STRIDE, INTERNAL, STRIDE, device=delta.device) / INTERNAL - 1.0
        gx, gy = torch.meshgrid(ndc, ndc, indexing="xy")
        gx = gx[None, None, None].expand(1, 1, LAYERS, h, w)
        gy = gy[None, None, None].expand(1, 1, LAYERS, h, w)

        # Means: tiny (x0.001) NDC offsets; depth through softplus(inverse_softplus(.) + .).
        zz = 1.0 / (F.softplus(_inverse_softplus(disp) + 0.001 * delta[:, 2:3]) + 1e-3)
        means = torch.cat(
            [zz * (gx + 0.001 * delta[:, 0:1]), zz * (gy + 0.001 * delta[:, 1:2]), zz], 1
        )

        # Scales: base proportional to depth, times 10 * sigmoid(.) (equal to 1 at delta = 0).
        base_scale = (1.0 / disp) * (2.0 * STRIDE / INTERNAL) * disp * zz
        a = 10.0 / (10.0 - 1.0)
        b = math.log(1.0 / 9.0)
        scales = base_scale * (10.0 * torch.sigmoid(a * delta[:, 3:6] + b))

        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=delta.device).view(1, 4, 1, 1, 1)
        quats = identity + delta[:, 6:10]
        colors = torch.sigmoid(
            _inverse_sigmoid(base_colors.clamp(0.01, 0.99)) + 0.1 * delta[:, 10:13]
        )
        opacities = torch.sigmoid(delta[:, 13])  # base opacity 0.5, i.e. logit 0

        def flat(t: Tensor) -> Tensor:  # (1, C, layers, h, w) -> (N, C), layer-major like Apple's
            return t[0].permute(1, 2, 3, 0).reshape(-1, t.shape[1])

        return Raw(
            means=flat(means) * global_scale,
            scales=flat(scales) * global_scale,
            quats=flat(quats),
            colors=flat(colors),
            opacities=opacities.reshape(-1),
        )


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_predictor(checkpoint: str | Path, device: str | torch.device = "cpu") -> Predictor:
    """Build the network and load the released checkpoint (memory-mapped, so no 2.8 GB copy)."""
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    state = {k: v for k, v in state.items() if not _UNUSED.search(k)}
    with torch.device("meta"):
        model = Predictor()
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().to(device)
