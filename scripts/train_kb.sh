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