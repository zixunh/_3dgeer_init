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
