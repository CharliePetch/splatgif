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

## What to expect

Measured on an i5-4570 (4 cores, 16 GB, no GPU):

| Step | Time | Memory |
|---|---|---|
| SHARP forward pass | about 2 min on that CPU (seconds on a GPU) | about 6 GB peak |
| Render, 640 px | about 0.25 s per frame | |
| Render, 1280 px | about 0.5 s per frame | |

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
