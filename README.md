# splatgif

One photo in, a looping 3D "wigglegram" GIF out.

Apple's [SHARP](https://github.com/apple/ml-sharp) predicts a 3D Gaussian splat from a single
image in one forward pass. `splatgif` puts a virtual camera exactly where the photo was taken,
swings it a few degrees around a point in the scene (the *focus*), renders every frame on the CPU
and writes one GIF. Frame 0 is your photograph; everything nearer or farther than the focus drifts
past it, like a Nishika lenticular camera.

```bash
pip install splatgif
splatgif photo.jpg --size 1280 --focus 0.5,0.4      # -> photo.gif
```

No GPU, Node, browser or compiler needed. Python 3.10+; PyTorch, NumPy, Pillow and numba are the
only dependencies.

## Install

```bash
pip install splatgif                 # JPEG, PNG, WebP, TIFF
pip install "splatgif[heic]"         # + iPhone HEIC photos
```

On Linux, `pip install torch` pulls the CUDA build (about 2.5 GB with its libraries). With no
NVIDIA GPU, install the small CPU build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install splatgif
```

### The model weights

The first run needs Apple's SHARP checkpoint (2.8 GB). It is **not** part of this package, and
Apple licenses it for **non-commercial research use only**
([terms](https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL), revocable). So splatgif asks
before downloading: interactively, or with `--accept-license` / `SPLATGIF_ACCEPT_LICENSE=1` in
scripts. The file is checksum-verified and cached where `torch.hub` keeps it
(`~/.cache/torch/hub/checkpoints/`), which is also where Apple's own `sharp` CLI puts it, so an
existing download is reused. To use a copy elsewhere: `--weights PATH` or `SPLATGIF_WEIGHTS=PATH`.

## Use

```bash
splatgif photo.jpg                                # 640 px, gentle sway
splatgif photo.jpg -o out.gif --size 1280         # bigger
splatgif photo.jpg --focus 0.7,0.6                # pivot on the subject (x,y as fractions)
splatgif photo.jpg --motion drastic               # +-8 degrees instead of +-2.5
splatgif photo.jpg --motion 4                     # or any peak yaw in degrees
```

| Option | Default | |
|---|---|---|
| `--size PX` | 640 | Long edge of the GIF. |
| `--focus X,Y` | centre | Pixel to pivot on, as fractions of width and height. |
| `--motion` | `moderate` | `moderate` (+-2.5 deg), `drastic` (+-8 deg) or degrees. |
| `--frames N` | 48 | Frames per loop. |
| `--delay MS` | 20 | Per-frame delay; 20 ms is the fastest browsers honour. |
| `--no-dither` | | Smaller file, more banding. |
| `--device` | best | `cuda`, `mps` or `cpu`. |

**Pass `--focus` whenever the subject is off-centre.** The default pivots on the depth at the
middle of the frame; if that is the background, the subject swings far out of frame and the result
looks like a shake rather than a wiggle. Pick the pixel on the nearest subject (a face, a belly, a
dog's nose) and let the background drift.

From Python:

```python
import splatgif

info = splatgif.make_gif("photo.jpg", "wiggle.gif", size=1280, focus=(0.7, 0.6))
print(info.focus_depth, info.predict_s, info.render_s)

# The slow step (SHARP) once, then as many renders as you like:
splat = splatgif.predict("photo.jpg")
frames = splat.render(size=640, motion="drastic", focus=(0.7, 0.6))   # list of PIL images
splatgif.save_gif(frames, "drastic.gif")
```

## Depth maps

The splat is also a depth map. `--depth FILE` measures the scene instead of animating it:

```bash
splatgif photo.jpg --depth depth.png --size 1280     # 16-bit PNG: metres x 256, 0 = no surface
splatgif photo.jpg --depth depth.npz                 # float32 metres, NaN = no surface
splatgif photo.jpg --depth look.png --depth-cmap     # a picture for eyes: red near, violet far
```

`--size` is the long edge, as for the GIF. Use the photo's own long edge and the map lines up with
the photo (and with frame 0 of the GIF) pixel for pixel, give or take rounding to even sizes. The
GIF-only options are ignored. The depth pass takes under a second at 1280 px once numba has compiled
(a few seconds on the very first run); the SHARP prediction before it costs what it costs for a GIF.

What the numbers are:

* **Metres, if the lens was right.** SHARP predicts metric depth, but only as well as it knows the
  focal length. A photo without EXIF lens data (screenshots, chat apps) is read as a 30 mm lens,
  and the scale is then approximate; the ordering and proportions are still right. splatgif says
  so when it happens, and the files record it.
* **z-depth**: distance along the optical axis, not from the camera centre.
* **The median depth of each pixel's ray**: the z where its accumulated opacity first reaches one
  half, using the same Gaussians in the same order as the colour render. Silhouettes stay sharp
  where an average would smear near and far together (on a test portrait, across 612 strong depth
  steps, the median 10 to 90 % transition was 3 px at 1280). A pixel the Gaussians never make half
  opaque is empty.

Files: `.npz` is lossless (float32 `depth`, plus `f_px` and `exif_focal`). `.png` and `.tif` are
16-bit grayscale in the KITTI convention (value = metres x 256, 0 = empty, saturating at 255.99 m,
resolution 0.004 m). A PNG also carries the focal length and the `exif_focal` flag in a
`splatgif-depth` text chunk. `--depth-cmap` writes a picture instead, scaled per image over
inverse depth; it holds no numbers and cannot be loaded back.

```python
splat = splatgif.predict("photo.jpg")                # the slow step, once
dm = splat.depth_map(1280)                           # DepthMap: .depth (h, w) float32, NaN = empty
splatgif.write_depth(dm, "depth.png")                # or .npz; cmap=True for a picture
dm = splatgif.load_depth("depth.png")                # back to float32 metres (within 0.002 m)
# camera-space position of pixel (u, v), width w and height h:
#   x = (u + .5 - w / 2) * z / dm.f_px,  y = (v + .5 - h / 2) * z / dm.f_px,  z = dm.depth[v, u]
```

**Distrust the border.** SHARP pulls its outermost Gaussians toward the camera. On a test portrait
the wall behind the subject is 1.5 to 1.9 m away, yet 95 % of the Gaussians within 32 px of the
frame edge (at 1280 px) sit nearer than 0.7 m. That is in the splat itself, not added by the depth
pass, so every map shows a band of about 30 px at 1280 (2.5 % of each side) that reads nearer than
the scene it frames. Crop it before using the numbers for anything.

## What to expect

Measured on an i5-4570 (4 cores, 16 GB, no GPU):

| Step | Time | Memory |
|---|---|---|
| SHARP forward pass | about 2 min on that CPU (seconds on a GPU) | about 6 GB peak |
| Render, 640 px | about 0.25 s per frame | |
| Render, 1280 px | about 0.5 s per frame | |
| Depth map, 1280 px | under 1 s | |

A 48-frame 1280 px GIF runs to tens of megabytes; convert to MP4 to share it
(`ffmpeg -stream_loop 4 -i in.gif -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" out.mp4`).
The first render in a fresh environment also pays a few seconds for numba to compile the
rasteriser; it is cached afterwards (set `NUMBA_CACHE_DIR` if the install directory is read-only).

Limits worth knowing:

* **No focal length in the photo's EXIF** (screenshots, messaging apps): a 30 mm lens is assumed.
  Frame 0 still matches the photo; only the depth scale is a guess. splatgif says so.
* **Foliage against sky** can show sky leaking through thin splats. That is the model, not a bug.
* `--motion drastic` can expose black voids at the frame edges on the swing side.
* Only the CPU path has been tested; CUDA and MPS run the same PyTorch code as Apple's CLI.

## How it works

1. **Photo and lens.** EXIF orientation is applied and the focal length converted to pixels.
2. **SHARP.** The photo is resampled to 1536 x 1536 and one forward pass of Apple's network
   predicts 1.18 million Gaussians seen from the photo's own camera. The network is re-written in
   `model.py` and loads the released checkpoint with `strict=True`; its output matches Apple's code
   to float32 noise.
3. **Pivot.** The focus point is the median depth of the Gaussians near the chosen pixel, on the
   ray through it.
4. **Sway.** Each frame yaws the camera about that point by `peak * sin(2 pi i / n)`, with the
   photo's own intrinsics, so frame 0 is the photograph and the pivot never moves.
5. **Render.** A numba rasteriser (EWA projection, depth sort, tile binning, front-to-back
   blending) draws each frame.
6. **GIF.** One 256-colour palette is built from pixels sampled across every frame (a palette per
   frame makes colours crawl), then each frame is Floyd-Steinberg dithered onto it.
7. **Depth (`--depth`).** Instead of steps 3 to 6, the same rasteriser walk records, per pixel, the
   z at which the ray turns half opaque.

## Development

```bash
pip install -e ".[dev]"
pytest                    # no weights needed
pytest -m slow            # + end-to-end with the real checkpoint (2.8 GB, a few minutes)
ruff check . && ruff format --check .
python -m build && twine check dist/*
```

## Licence

splatgif is MIT-licensed ([LICENSE](LICENSE)). `src/splatgif/model.py` is a condensed
re-implementation of the network in Apple's ml-sharp, which stays under Apple's own software
licence ([licenses/APPLE-ML-SHARP-LICENSE](licenses/APPLE-ML-SHARP-LICENSE)). The SHARP weights are
research-only ([licenses/APPLE-ML-SHARP-LICENSE_MODEL](licenses/APPLE-ML-SHARP-LICENSE_MODEL)), so
do not use the GIFs commercially.

## Credit

SHARP is by Lars Mescheder, Wei Dong, Shiwei Li, Xuyang Bai, Marcel Santos, Peiyun Hu, Bruno
Lecouat, Mingmin Zhen, Amael Delaunoy, Tian Fang, Yanghai Tsin, Stephan R. Richter and Vladlen
Koltun: *Sharp Monocular View Synthesis in Less Than a Second*,
[arXiv:2512.10685](https://arxiv.org/abs/2512.10685). Its sliding-pyramid encoder comes from
Depth Pro.
