set -e
SKIP_TRAIN=true

SCENE_ID=4ef75031e3 #2a1a3afad9 1f7cbbdde1 4ef75031e3 1d003b07bd 0a5c013435
DATA_ROOT=/media/scannetpp/demo/
DATASET_DIR=$DATA_ROOT$SCENE_ID/dslr/
OUTPUT_DIR=/backup/omni-3dgs/output_achive/scannetpp/$SCENE_ID

STEP_EVAL=0.002
FOVMOD_EVAL=2.0
FOVMAP_DIR_EVAL=undistorted_fovmaps_fov_"$FOVMOD_EVAL"_step_"$STEP_EVAL"/
TEST_MASK_FN=fov_"$FOVMOD_EVAL"_step_"$STEP_EVAL"_mask.png

DIST_SCALING=0.0
FOCAL_SCALING=0.75
MIRR_SHIFT=0.0
RENDER_MODEL=KB

ITERS_NUM=30000

# eval
python data/scannetpp/scnt_raymap.py --path $DATA_ROOT \
                              --scenes $SCENE_ID \
                              --distortion_scaling $DIST_SCALING \
                              --focal_scaling $FOCAL_SCALING \
                              --mirror_shift $MIRR_SHIFT

# render
python render.py \
    -m $OUTPUT_DIR \
    -s $DATASET_DIR \
    --iteration $ITERS_NUM \
    --camera_model FISHEYE \
    --render_model $RENDER_MODEL \
    --skip_train \
    --sample_step $STEP_EVAL --fov_mod $FOVMOD_EVAL \
    --train_test_exp \
    --raymap_path "$DATASET_DIR"raymap_fisheye.npy \
    --distortion_scaling $DIST_SCALING \
    --focal_scaling $FOCAL_SCALING \
    --mirror_shift $MIRR_SHIFT