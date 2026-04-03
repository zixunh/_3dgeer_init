#!/usr/bin/env bash
set -e

SCENE_ID=$1
DATA_ROOT=$2
CKPT_DIR=$3
MODE=$4 # KB, BEAP, or PH
ITERS_NUM=30000
STEP_EVAL=0.002
FOVMOD_EVAL=2.0

DATASET_DIR=$2/$1
OUTPUT_DIR=$3/$1

echo "Scene: $SCENE_ID"
echo "Mode: $MODE"

if [ "$MODE" = "BEAP" ]; then

BEAP_DIR_EVAL="beap_fov_${FOVMOD_EVAL}_step_${STEP_EVAL}/"
EVAL_MASK_FN="fov_${FOVMOD_EVAL}_step_${STEP_EVAL}_mask.png"
python data/scnt/scnt_kb2beap.py --path "$DATASET_DIR" --dst "$BEAP_DIR_EVAL" --step "$STEP_EVAL" --fov_mod "$FOVMOD_EVAL" --mask_dst "$EVAL_MASK_FN"

echo "Rendering BEAP space"

python render.py \
    -m ${OUTPUT_DIR} \
    -s ${DATASET_DIR} \
    --iteration ${ITERS_NUM} \
    --camera_model FISHEYE \
    --render_model ${MODE} \
    --skip_train \
    --sample_step ${STEP_EVAL} \
    --fov_mod ${FOVMOD_EVAL} \
    --train_test_exp


elif [ "$MODE" = "KB" ]; then

echo "Rendering KB fisheye"

DIST_SCALING=1.0 # set DIST_SCALING as 0 to render EQ
FOCAL_SCALING=1.0
MIRR_SHIFT=0.0

echo "== Generating ray map =="
python data/scnt/scnt_raymap.py \
    --path ${DATA_ROOT} \
    --scenes ${SCENE_ID} \
    --distortion_scaling ${DIST_SCALING} \
    --focal_scaling ${FOCAL_SCALING} \
    --mirror_shift ${MIRR_SHIFT}

echo "== Rendering =="
python render.py \
    -m ${OUTPUT_DIR} \
    -s ${DATASET_DIR} \
    --iteration ${ITERS_NUM} \
    --camera_model FISHEYE \
    --render_model ${MODE} \
    --skip_train \
    --sample_step ${STEP_EVAL} \
    --fov_mod ${FOVMOD_EVAL} \
    --train_test_exp \
    --raymap_path ${DATASET_DIR}/raymap_fisheye.npy \
    --distortion_scaling ${DIST_SCALING} \
    --focal_scaling ${FOCAL_SCALING} \
    --mirror_shift ${MIRR_SHIFT}

elif [ "$MODE" = "PH" ]; then

echo "Rendering Pinhole"

FOCAL_SCALING=1.0

echo "== Rendering =="
python render.py \
    -m ${OUTPUT_DIR} \
    -s ${DATASET_DIR} \
    --iteration ${ITERS_NUM} \
    --camera_model FISHEYE \
    --render_model ${MODE} \
    --skip_train \
    --sample_step ${STEP_EVAL} \
    --fov_mod ${FOVMOD_EVAL} \
    --train_test_exp \
    --focal_scaling ${FOCAL_SCALING}

else

echo "No specific MODE given — rendering all three modes: BEAP, KB, PH"

# ---- BEAP ----
BEAP_DIR_EVAL="beap_fov_${FOVMOD_EVAL}_step_${STEP_EVAL}/"
EVAL_MASK_FN="fov_${FOVMOD_EVAL}_step_${STEP_EVAL}_mask.png"
python data/scnt/scnt_kb2beap.py --path "$DATASET_DIR" --dst "$BEAP_DIR_EVAL" --step "$STEP_EVAL" --fov_mod "$FOVMOD_EVAL" --mask_dst "$EVAL_MASK_FN"

echo "Rendering BEAP space"
python render.py \
    -m ${OUTPUT_DIR} \
    -s ${DATASET_DIR} \
    --iteration ${ITERS_NUM} \
    --camera_model FISHEYE \
    --render_model BEAP \
    --skip_train \
    --sample_step ${STEP_EVAL} \
    --fov_mod ${FOVMOD_EVAL} \
    --train_test_exp

cp -r "${OUTPUT_DIR}/test/ours_${ITERS_NUM}/renders" "${OUTPUT_DIR}/renders_BEAP"

# ---- KB ----
echo "Rendering KB fisheye"
DIST_SCALING=1.0
FOCAL_SCALING=1.0
MIRR_SHIFT=0.0

echo "== Generating ray map =="
python data/scnt/scnt_raymap.py \
    --path ${DATA_ROOT} \
    --scenes ${SCENE_ID} \
    --distortion_scaling ${DIST_SCALING} \
    --focal_scaling ${FOCAL_SCALING} \
    --mirror_shift ${MIRR_SHIFT}

echo "== Rendering =="
python render.py \
    -m ${OUTPUT_DIR} \
    -s ${DATASET_DIR} \
    --iteration ${ITERS_NUM} \
    --camera_model FISHEYE \
    --render_model KB \
    --skip_train \
    --sample_step ${STEP_EVAL} \
    --fov_mod ${FOVMOD_EVAL} \
    --train_test_exp \
    --raymap_path ${DATASET_DIR}/raymap_fisheye.npy \
    --distortion_scaling ${DIST_SCALING} \
    --focal_scaling ${FOCAL_SCALING} \
    --mirror_shift ${MIRR_SHIFT}

cp -r "${OUTPUT_DIR}/test/ours_${ITERS_NUM}/renders" "${OUTPUT_DIR}/renders_KB"

# ---- PH ----
echo "Rendering Pinhole"
FOCAL_SCALING=1.0

echo "== Rendering =="
python render.py \
    -m ${OUTPUT_DIR} \
    -s ${DATASET_DIR} \
    --iteration ${ITERS_NUM} \
    --camera_model FISHEYE \
    --render_model PH \
    --skip_train \
    --sample_step ${STEP_EVAL} \
    --fov_mod ${FOVMOD_EVAL} \
    --train_test_exp \
    --focal_scaling ${FOCAL_SCALING}

cp -r "${OUTPUT_DIR}/test/ours_${ITERS_NUM}/renders" "${OUTPUT_DIR}/renders_PH"

# ---- Stitch all renders into one comparison image with matplotlib ----
echo "Stitching renders with matplotlib..."
python3 - "$OUTPUT_DIR" <<'PYEOF'
import sys, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

output_dir = sys.argv[1]
modes = ["BEAP", "KB", "PH"]

mode_imgs = {}
for mode in modes:
    d = os.path.join(output_dir, f"renders_{mode}")
    mode_imgs[mode] = sorted(glob.glob(os.path.join(d, "*.png")))

n = min(len(v) for v in mode_imgs.values())
if n == 0:
    print("No rendered images found in mode directories.", file=sys.stderr)
    sys.exit(1)

fig, axes = plt.subplots(n, 3, figsize=(18, 6 * n))
if n == 1:
    axes = [axes]

for i in range(n):
    for j, mode in enumerate(modes):
        img = mpimg.imread(mode_imgs[mode][i])
        axes[i][j].imshow(img)
        axes[i][j].set_title(f"{mode} — {os.path.basename(mode_imgs[mode][i])}")
        axes[i][j].axis("off")

plt.tight_layout()
out_path = os.path.join(output_dir, "renders_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved comparison image: {out_path}")
PYEOF

fi