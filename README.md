<div align="center">
<h1>TetraGEER</h1>
<h3>Minimal geometrically identifiable structure for under-constrained 3D radiance fields</h3>

Built on top of <strong>3DGEER</strong>: exact and efficient ray-Gaussian rendering for generic cameras.

<p align="center">
  <img src="assets/teaser.gif" alt="3DGEER teaser" style="width: 100%;">
</p>
</div>

## Overview

TetraGEER extends the existing 3DGEER codebase with a tetrahedral geometry carrier. Instead of optimizing free Gaussian centers, scales, and rotations directly, TetraGEER initializes a tetrahedral structure from the input point cloud, optimizes shared tetra vertices, and derives Gaussian primitives from each tetrahedron by moment matching.

The current implementation is a dual-mode codebase:

- `--model_type gaussian`: original free-Gaussian 3DGEER behavior.
- `--model_type tetra`: TetraGEER path with Delaunay tetra initialization and PyTorch tetra-to-Gaussian reparameterization.

The tetra path currently reuses the existing GEER CUDA renderer after deriving Gaussian means, scales, and rotations in PyTorch. Native tetra CUDA entrypoints are scaffolded in `submodules/geer-rasterizer`, but the active training path is the parity implementation.

## Key Features

- Delaunay tetrahedral initialization from COLMAP or scene point clouds.
- Shared tetra vertex optimization with fixed tetra connectivity for the first working version.
- Analytic tetra-to-Gaussian moment matching:
  - mean from tetra vertex average
  - covariance from centered tetra vertices
  - scale and rotation from covariance eigendecomposition
- Per-tetra opacity and spherical-harmonic appearance.
- Geometry regularizers for Laplacian smoothing, ARAP-style edge preservation, volume barriers, and minimality.
- Compatibility export to `point_cloud.ply` so existing render/view tools can still consume derived Gaussians.

## Relation to 3DGEER

This repository still contains the 3DGEER renderer and scripts. TetraGEER changes the scene representation, not the default rendering backend. The original 3DGEER project resources remain useful for understanding the GEER CUDA renderer:

- [3DGEER arXiv](https://arxiv.org/abs/2505.24053)
- [3DGEER project page](https://zixunh.github.io/3d-geer/)
- [GEER CUDA rasterizer](./submodules/geer-rasterizer/)

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">📰BibTeX</h2>
    <p>If you find our work useful, we’d really appreciate a ⭐ or citation.</p>
    <pre><code>@misc{huang20263dgeer3dgaussianrendering,
      title={3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras}, 
      author={Zixun Huang and Cho-Ying Wu and Yuliang Guo and Xinyu Huang and Liu Ren},
      year={2026},
      eprint={2505.24053},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2505.24053}, 
}</code></pre>
  </div>
</section>


## 🎉News
<p align="center">
  <img src="assets/drive-geer.gif" alt="teaser" style="width: 100%;">
</p>

- **TBD**: `drivestudio-geer` and `stormGaussian-geer` will be released [here](#special-extension) as well!
- **2026-03-19**: 3DGEER now supports **dynamic outdoor scene** rendering under **wide-FoV fisheye** cameras with the integration into DriveStudio.
- **2026-03-17**: `SIBR_remoteGaussian_app` is adapted to our work as an interactive viewer for training and trained checkpoints. Try the `BEAP` mode in the viewer; `Pinhole` and `Fisheye` modes are supported as well.
- **2026-03-09**: `gsplat-geer` released [here](#special-extension)!
- **2026-03-09**: Code released! Can Gaussian rendering be both exact and fast without relying on lossy splatting? **Check out 3DGEER**!
- **2026-03-09**: Code release approved. License updated. Requested admin to push code to BoschResearch.
- **2026-01-25**: 3DGEER accepted to **ICLR 2026**, with an [initial review](https://openreview.net/forum?id=4voMNlRWI7) of average **7** (top 1% score).
- **2025-05-29**: Preprint released on [Arxiv](https://arxiv.org/abs/2505.24053).


## 📷3DGEER-CUDA-Rasterizer
The full CUDA implementation can be found here: [./submodules/geer-rasterizer/](./submodules/geer-rasterizer/).
#### Key Insight 1: Fixing the Math Behind Gaussian Projection

- Ray–Gaussian Integral (Forward & Backward): Analytical forward rendering and numerical stable backward gradient computation. (See [paper](https://arxiv.org/pdf/2505.24053) Appendix C for the math.)

  <div align="center">
    <img src="assets/forward2backward.gif" width="60%">
  </div>

#### Key Insight 2: Fixing the Math Behind Gaussian Association

- Particle Bounding Frustum: Exact and minimal boundary geometry for ray–particle association. (See [paper](https://arxiv.org/pdf/2505.24053) Appendix D for the math.)

  <div align="center">
    <img src="assets/asso.gif" width="60%">
  </div>

#### Key Insight 3: Optimizing the Ray Distribution Behind Pixelwise Color Supervision

- Bipolar Equiangular Projection: Maintains uniform ray sampling across arbitrary fields of view, thereby providing stable, FoV-invariant supervision for radiance field training.

<div align="center">
  <img src="assets/beap.gif" width="60%">
</div>

## Installation

TetraGEER uses the original 3DGEER/3DGS Python stack plus SciPy for CPU Delaunay initialization.
Run these commands from the `tetrageer` directory.

```sh
conda env create -f environment.yml
conda activate gaussian_splatting
pip install --no-build-isolation ./submodules/geer-rasterizer
```

The environment file includes `scipy`. If you are using an existing 3DGEER environment, install it manually:

```sh
pip install scipy
```

### Docker

Set your data path and codebase path in `./docker/init_my_docker.sh`, then build the matching image:

```sh
bash ./docker/build.sh 4090
bash ./docker/init_my_docker.sh
```

After modifying the GEER rasterizer, rebuild the extension inside the container:

```sh
pip install --no-build-isolation ./submodules/geer-rasterizer
```

## 🔧Interactive Viewer Setup
<div align="center">
  <img src="assets/demo_sibr.gif" width="100%">
</div>

#### SIBR Viewer Configuration with Docker 
**⚠️ Important Notice on Visualization:**
`SIBR_gaussianViewer_app` is currently not supported for Gaussian Exact and Efficient Rendering (GEER).
Please do not use:
```sh
  $sibr_gv -m "./output/scnt/<SCENE_ID>/dslr"
```
as it invokes the **vanilla 3D Gaussian Splatting rasterizer** for offline rendering from checkpoints. This leads to invalid results, since GEER-trained (ray-based) scenes are incompatible with splatting-based rendering.

**✅ Recommended Alternatives:**
- **During training**, use `SIBR_remoteGaussian_app`, which connects via port and calls our modified GEER rasterizer.
```sh
# Enter Workspace for SIBR Viewer on Terminal 2
bash ./docker/run_my_docker.sh
# Inside docker container, run:
$sibr_rg
```
- To use `SIBR_remoteGaussian_app` from **checkpoints**, first run the following on Terminal 1.
```sh
bash scripts/visualize.sh <SCENE_ID> <DATA_ROOT> <CKPT_DIR> <MODE: BEAP, PH or KB> # Example: 1d003b07bd/dslr data/scnt/datasets ckpt/scnt KB
# Then (on Terminal 2) launch:
$sibr_rg
```
- **For offline visualization**, We recommend using our [`gsplat-geer`](https://github.com/boschresearch/3dgeer/tree/gsplat-geer) implementation, built on top of: https://github.com/nerfstudio-project/gsplat/blob/main/docs/3dgut.md 
> Note: the mismatched culling issue in UT is resolved using our PBF-based fix.


## Quick Start

### 1. Prepare data

TetraGEER uses the same scene loaders as 3DGEER. A COLMAP-style scene should contain images and sparse points:

```text
<scene>
  images/
  sparse/0/
    cameras.bin or cameras.txt
    images.bin or images.txt
    points3D.bin or points3D.txt
```

The point cloud is used twice in tetra mode: first as the scene initialization source, and then as input to CPU Delaunay tetrahedralization.

### 2. Train TetraGEER

Run from the `tetrageer` directory:

```sh
python train.py \
  -s <path/to/scene> \
  -m <path/to/output> \
  --model_type tetra \
  --tetra_init delaunay \
  --tetra_downsample_voxel 0.02 \
  --render_model BEAP \
  --iterations 30000
```

For a small smoke run:

```sh
python train.py \
  -s <path/to/scene> \
  -m ./output/tetrageer_smoke \
  --model_type tetra \
  --tetra_downsample_voxel 0.05 \
  --iterations 1000 \
  --disable_viewer
```

Important tetra arguments:

| Argument | Description | Default |
|---|---|---|
| `--model_type` | `gaussian` for original 3DGEER, `tetra` for TetraGEER | `gaussian` |
| `--tetra_init` | Tetra initialization method. Currently supports `delaunay` | `delaunay` |
| `--tetra_downsample_voxel` | Voxel size for point-cloud downsampling before Delaunay. Use larger values for faster initialization | `0.0` |
| `--tetra_eta` | Covariance scale for tetra-to-Gaussian moment matching | `1.0` |
| `--tetra_eps` | Covariance diagonal stabilizer | `1e-4` |
| `--tetra_lap_weight` | Laplacian edge smoothing weight | `0.0` |
| `--tetra_arap_weight` | Edge-length preservation weight | `0.0` |
| `--tetra_vol_weight` | Volume barrier weight for inverted/collapsed tetrahedra | `0.0` |
| `--tetra_min_weight` | Minimality penalty on active tetra opacity/volume | `0.0` |

Camera/rendering arguments are inherited from 3DGEER:

| Argument | Description | Default |
|---|---|---|
| `--render_model` | Projection mode: `BEAP`, `KB`, `EQ`, or `PH` | `BEAP` |
| `--sample_step` | Ray sampling interval for BEAP/KB modes | scene/script dependent |
| `--raymap_path` | Per-pixel ray-direction map for KB/EQ mode | `None` |
| `--mask_path` | Optional validity mask | `None` |

### 3. Inspect outputs

Each saved iteration writes:

```text
<output>/point_cloud/iteration_<N>/
  tetra_state.npz   # native tetra vertices, connectivity, opacity, SH features
  point_cloud.ply   # derived Gaussian compatibility export
```

Use `tetra_state.npz` for TetraGEER-aware training/resume/rendering. Use `point_cloud.ply` when you need compatibility with tools that expect Gaussian checkpoints.

### 4. Render a trained TetraGEER model

Render with the same model type:

```sh
python render.py \
  -m <path/to/output> \
  -s <path/to/scene> \
  --model_type tetra \
  --iteration -1 \
  --render_model BEAP
```

To render the compatibility Gaussian export instead, use `--model_type gaussian`; this loads `point_cloud.ply` and bypasses the native tetra state.

### 5. Original 3DGEER mode

The original free-Gaussian path is unchanged:

```sh
python train.py \
  -s <path/to/scene> \
  -m <path/to/output> \
  --model_type gaussian \
  --render_model BEAP
```

The existing script wrappers under [scripts](./scripts) still target the original 3DGEER workflow unless you add `--model_type tetra` and tetra options.

## Available 3DGEER Checkpoints

The original 3DGEER checkpoints remain useful for renderer comparisons:

- [ScanNet++: Kitchen, Lab, Officeroom, Bedroom, Storage](https://www.dropbox.com/scl/fi/fk28mxew8xt8qpj4mi5ch/scannetpp.zip?rlkey=lcg7g3mvvdw7351ocs1sdxfc2&e=1&st=0skeao82&dl=0)
- [ZipNeRF: Alameda, Berlin, London, NYC](https://www.dropbox.com/scl/fo/3bbsmmrqkno774e672p7k/ADulNsSAaCZ2RVBQjWlECb0?rlkey=ugdqyf3cja9l8v29jhrbw6ute&e=1&st=t5ht8jmt&dl=0)
- [Aria: Livingroom, Steakhouse, Garden](https://www.dropbox.com/scl/fo/u3vqri9u0799t8w4xqr27/APNPKlXKih3MQpUoU2jIHAI?rlkey=mria3secffbcaxqwuk3hg2248&st=yk4x06y3&dl=0)
- [Tank and Temples: Train, Truck](https://www.dropbox.com/scl/fo/gl4wwzbgwf781o0n836hw/AFONH7H2XsADaJoGPMqRIKY?rlkey=jc5r2jkmzko6cmrs5ax0d5y2r&e=1&st=frfpi268&dl=0)

## 🙏Special Extension
<p align="center">
  <a href='https://github.com/boschresearch/3dgeer/tree/gsplat-geer'><img src="assets/gsplat-geer.gif" alt="teaser" style="width: 100%;"></a>
  3DGEER supports the opensource community with <code>gsplat</code> integration. <br />Check out our <a href='https://github.com/boschresearch/3dgeer/tree/gsplat-geer'><code>gsplat-geer</code></a> branch for details.
</p>

## ⛽️Contributing
Feel free to drop a [pull request](https://github.com/boschresearch/3dgeer/pulls) whenever!

## 👀Visuals ([More](https://zixunh.github.io/3d-geer))

<!-- <div class="col-md-8 col-md-offset-2">
    <h4>
      Out-of-Distribution (OOD) Extreme FoV Results
    </h4>
    <div align="center">
      <img src="assets/our_gif/gut_render.gif" width="47%" style="padding: 2px;">
      <img src="assets/our_gif/geer_render.gif" width="47%" style="padding: 2px;">
    </div>
    <div align="center">
      <img src="assets/our_gif/gut_train_gut_render.gif" width="47%" style="padding: 2px;">
      <img src="assets/our_gif/gut_train_geer_render.gif" width="47%" style="padding: 2px;">
    </div>
    <p align="center">
      Side-by-Side Comparison on Close-Up Parking Data: 3DGEER's PBF association (Right Col.) has less popping issues (First Row) and no grid-line artifacts (Second Row) compared w/ UT (Left Col.).
    </p>
</div> -->

<div class="col-md-8 col-md-offset-2">
    <h4>
        High-Quality Large FoV Results
    </h4>
    <div align="center">
      <img src="assets/our_gif/alameda3_web.gif" width="32%" style="padding: 2px;">
      <img src="assets/our_gif/alameda1_web.gif" width="32%" style="padding: 2px;">
      <img src="assets/our_gif/alameda2_web.gif" width="32%" style="padding: 2px;">
    </div>
</div>
<div class="col-md-8 col-md-offset-2">
    <h4>
        Highly Distorted & Close-Up Views
    </h4>
    <div align="center">
      <img src="assets/our_gif/berlin4_web.gif" width="32%" style="padding: 2px;">
      <img src="assets/our_gif/berlin8_web.gif" width="32%" style="padding: 2px;">
      <img src="assets/our_gif/nyc2_web.gif" width="32%" style="padding: 2px;">
    </div>
</div>


## 💡License
3DGEER is released under the AGPL-3.0 License. See the [LICENSE](./LICENSE.md) file for details.
This project is built upon [3D Gaussian Splatting by Inria](https://github.com/graphdeco-inria/gaussian-splatting). We thank the authors for their excellent open-source work. The original license and copyright notice are included in this repository, see the file [3dgs-license.txt](./3dgs-license.txt).
