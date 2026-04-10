# Train and Eval Documentation for 3DGEER
### Train Examples

**Arguments:**

`SCENE_ID` : scene name (e.g. `1d003b07bd/dslr`, `steakhouse`, `truck`)

`DATA_ROOT` : root directory of the formatted dataset (e.g. `data/scnt/datasets`)

`OUTPUT_DIR` : directory where the trained model will be saved (e.g. `output/scnt/1d003b07bd/dslr`)

`STEP` : ray sampling interval in radians for BEAP/KB training (e.g. `0.002`)

`FOVMOD_TRAIN` : FoV scale factor applied during training (e.g. `1.3`)

---

#### ScanNet++ — BEAP Mode (default, recommended)

Use the convenience script to train on multiple scenes at once:

```bash
bash scripts/train.sh
```

The script trains on `1d003b07bd/dslr` and `1f7cbbdde1/dslr` by default. To train on a single scene manually:

```bash
SCENE_ID=1d003b07bd/dslr
DATA_ROOT=data/scnt/datasets
OUTPUT_DIR=output/scnt/${SCENE_ID}
STEP=0.002
FOVMOD_TRAIN=1.3

DATASET_DIR=${DATA_ROOT}/${SCENE_ID}
BEAP_DIR_TRAIN=beap_fov_${FOVMOD_TRAIN}_step_${STEP}
TRAIN_MASK_FN=fov_${FOVMOD_TRAIN}_step_${STEP}_mask.png

# Step 1 – Convert fisheye images to BEAP equiangular space
python data/scnt/scnt_kb2beap.py \
    --path ${DATASET_DIR} --dst ${BEAP_DIR_TRAIN} \
    --step ${STEP} --fov_mod ${FOVMOD_TRAIN} \
    --mask_dst ${TRAIN_MASK_FN}

# Step 2 – Train
python train.py \
    -s ${DATASET_DIR} -m ${OUTPUT_DIR} \
    --iterations 30000 \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 30000 \
    --resolution 1 --eval \
    --sample_step ${STEP} --fov_mod ${FOVMOD_TRAIN} \
    --mask_path ${DATASET_DIR}/${BEAP_DIR_TRAIN}/${TRAIN_MASK_FN} \
    --sibr_mask_refcam ${DATASET_DIR}/colmap/cameras_fish.txt
```

---

#### ScanNet++ — KB Mode (Kannala-Brandt fisheye)

```bash
SCENE_ID=1d003b07bd/dslr
DATA_ROOT=data/scnt/datasets
OUTPUT_DIR=output/scnt/${SCENE_ID}
STEP=0.002
FOVMOD_TRAIN=1.3
DIST_SCALING=1.0   # set to 0 for equidistant (EQ) mode
FOCAL_SCALING=1.0
MIRR_SHIFT=0.0

DATASET_DIR=${DATA_ROOT}/${SCENE_ID}

# Step 1 – Generate per-pixel ray map
python data/scnt/scnt_raymap.py \
    --path ${DATA_ROOT} --scenes ${SCENE_ID} \
    --distortion_scaling ${DIST_SCALING} \
    --focal_scaling ${FOCAL_SCALING} \
    --mirror_shift ${MIRR_SHIFT}

# Step 2 – Train in KB mode
python train.py \
    -s ${DATASET_DIR} -m ${OUTPUT_DIR} \
    --iterations 30000 \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 30000 \
    --resolution 1 --eval \
    --render_model KB \
    --raymap_path ${DATASET_DIR}/raymap_fisheye.npy \
    --focal_scaling ${FOCAL_SCALING} \
    --distortion_scaling ${DIST_SCALING} \
    --mirror_shift ${MIRR_SHIFT} \
    --sample_step ${STEP} --fov_mod ${FOVMOD_TRAIN}
```

---

#### Tank and Temples — PH Mode (pinhole)

```bash
SCENE_ID=truck
DATA_ROOT=data/tt/datasets
OUTPUT_DIR=output/tt/${SCENE_ID}

DATASET_DIR=${DATA_ROOT}/${SCENE_ID}

python train.py \
    -s ${DATASET_DIR} -m ${OUTPUT_DIR} \
    --iterations 30000 \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 30000 \
    --resolution 1 --eval \
    --render_model PH \
    --dataset COLMAP \
    --camera_model PINHOLE \
    --densify_grad_threshold 0.002
```
### Render and Eval Examples
ScanNet++ Dataset
```bash
bash scripts/render.sh 1d003b07bd/dslr data/scnt/datasets ckpt/scnt KB
bash scripts/eval.sh 1d003b07bd/dslr data/scnt/datasets ckpt/scnt KB
bash scripts/eval.sh 1d003b07bd/dslr data/scnt/datasets ckpt/scnt BEAP
```
Aria Dataset
```bash
bash scripts/render.sh steakhouse data/aria/scannetpp_formatted ckpt/aria KB
bash scripts/eval.sh steakhouse data/aria/scannetpp_formatted ckpt/aria KB
bash scripts/eval.sh steakhouse data/aria/scannetpp_formatted ckpt/aria BEAP
```
Tanks and Temples
```bash
bash scripts/render.sh truck data/tt/datasets ckpt/tt PH
bash scripts/eval.sh truck data/tt/datasets ckpt/tt PH
bash scripts/eval.sh truck data/tt/datasets ckpt/tt BEAP
```
### Asso Mode Ablation
The `--asso_mode` argument controls the Gaussian association (tile culling) method used during rendering. Three modes are supported:

| `asso_mode` | Method | Description |
|:-----------:|--------|-------------|
| `0` | **PBF** (default) | Particle Bounding Frustum — exact and tight association |
| `1` | **EWA** | AABB via Elliptical Weighted Average |
| `2` | **UT** | AABB via Unscented Transform |

To run the ablation, pass `--asso_mode <value>` directly to `render.py`. Example using ScanNet++ with KB mode:
```bash
# PBF (default, asso_mode=0)
python render.py -m ckpt/scnt/1d003b07bd/dslr -s data/scnt/datasets/1d003b07bd/dslr \
    --iteration 30000 --camera_model FISHEYE --render_model KB --skip_train \
    --sample_step 0.0015 --fov_mod 2.0 --train_test_exp \
    --raymap_path data/scnt/datasets/1d003b07bd/dslr/raymap_fisheye.npy \
    --asso_mode 0

# EWA (asso_mode=1)
python render.py -m ckpt/scnt/1d003b07bd/dslr -s data/scnt/datasets/1d003b07bd/dslr \
    --iteration 30000 --camera_model FISHEYE --render_model KB --skip_train \
    --sample_step 0.0015 --fov_mod 2.0 --train_test_exp \
    --raymap_path data/scnt/datasets/1d003b07bd/dslr/raymap_fisheye.npy \
    --asso_mode 1

# UT (asso_mode=2)
python render.py -m ckpt/scnt/1d003b07bd/dslr -s data/scnt/datasets/1d003b07bd/dslr \
    --iteration 30000 --camera_model FISHEYE --render_model KB --skip_train \
    --sample_step 0.0015 --fov_mod 2.0 --train_test_exp \
    --raymap_path data/scnt/datasets/1d003b07bd/dslr/raymap_fisheye.npy \
    --asso_mode 2
```

### Evaluation Protocol

Unlike Fisheye-GS and 3DGUT, which remap the ScanNet++ ground-truth images into an equidistant projection space for evaluation, we perform evaluation directly under the original camera model (KB) of the dataset.

For methods using different projection models (e.g., equidistant, pinhole, or BEAP), the rendered outputs must be remapped back to the original camera space before comparison with the ground truth.

However, such remapping inevitably introduces pixel-level interpolation shifts, which can distort perceptual metrics (e.g., PSNR / SSIM / LPIPS) and prevent them from faithfully reflecting the actual scene reconstruction quality.

To ensure pixel-wise alignment, we apply the same remapping operation to the ground-truth images as well. This guarantees that both predictions and ground truth undergo identical interpolation, eliminating evaluation bias caused by projection conversion.

For the ray mapping used in this process (to reproduce the numbers in Table K.2), we use the grid map provided by:

https://huggingface.co/yuliangguo/depth-any-camera/blob/main/scannetpp_dac_swinl_indoor_2025_01.zip

Or run:
```python
  python data/scnt/scnt_raymap_dac.py 
```

Qualitative comparisons can be found in Figure K.4.

<p align="center">
  <img src="../assets/scnt_comp.jpg" alt="teaser" style="width: 100%;">
  Figure K.4.; https://arxiv.org/abs/2505.24053
</p>