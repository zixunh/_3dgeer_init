#!/usr/bin/env bash
# visualize.sh – launch the online SIBR visualizer for a trained 3DGEER model.
#
# Usage:
#   bash visualize.sh [MODE]
#
#   MODE  Render mode: BEAP (default), KB, or PH
#
# Edit the variables below to match your scene and output paths.

set -e

# ------------------------------------------------------------------
# User-configurable settings
# ------------------------------------------------------------------

SCENE_ID="1d003b07bd/dslr"
DATA_ROOT="data/scnt/datasets/"

# Must match the values used during training
STEP=0.002
FOVMOD_VIS=1

# Iteration to load (-1 = latest checkpoint)
ITER=30000

# Network GUI settings (must match what the SIBR viewer connects to)
IP="127.0.0.1"
PORT=6009

# KB / PH mode scalings
DIST_SCALING=1.0
FOCAL_SCALING=1.0
MIRR_SHIFT=0.0

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------

MODE="${1:-BEAP}"

echo "Visualizing scene: $SCENE_ID"
echo "Mode: $MODE"

DATASET_DIR="${DATA_ROOT}${SCENE_ID}/"
OUTPUT_DIR="output/scnt/${SCENE_ID}"

if [ "$MODE" = "BEAP" ]; then

    BEAP_DIR_VIS="beap_fov_${FOVMOD_VIS}_step_${STEP}/"
    TRAIN_MASK_FN="fov_${FOVMOD_VIS}_step_${STEP}_mask.png"

    python visualizer.py \
        -s "$DATASET_DIR" \
        -m "$OUTPUT_DIR" \
        --iteration "$ITER" \
        --ip "$IP" \
        --port "$PORT" \
        --render_model "$MODE" \
        --sample_step "$STEP" \
        --fov_mod "$FOVMOD_VIS" \
        --mask_path "${DATASET_DIR}${BEAP_DIR_VIS}${TRAIN_MASK_FN}" \
        --sibr_mask_refcam "${DATASET_DIR}colmap/cameras_fish.txt"

elif [ "$MODE" = "KB" ]; then

    echo "== Generating ray map =="
    python data/scnt/scnt_raymap.py \
        --path "$DATA_ROOT" \
        --scenes "$SCENE_ID" \
        --distortion_scaling "$DIST_SCALING" \
        --focal_scaling "$FOCAL_SCALING" \
        --mirror_shift "$MIRR_SHIFT"

    python visualizer.py \
        -s "$DATASET_DIR" \
        -m "$OUTPUT_DIR" \
        --iteration "$ITER" \
        --ip "$IP" \
        --port "$PORT" \
        --render_model "$MODE" \
        --sample_step "$STEP" \
        --fov_mod "$FOVMOD_VIS" \
        --raymap_path "${DATASET_DIR}raymap_fisheye.npy" \
        --distortion_scaling "$DIST_SCALING" \
        --focal_scaling "$FOCAL_SCALING" \
        --mirror_shift "$MIRR_SHIFT"

elif [ "$MODE" = "PH" ]; then

    python visualizer.py \
        -s "$DATASET_DIR" \
        -m "$OUTPUT_DIR" \
        --iteration "$ITER" \
        --ip "$IP" \
        --port "$PORT" \
        --render_model "$MODE" \
        --sample_step "$STEP" \
        --fov_mod "$FOVMOD_VIS" \
        --focal_scaling "$FOCAL_SCALING"

else
    echo "Unknown MODE: $MODE"
    echo "Valid modes are: BEAP, KB, PH"
    exit 1
fi
