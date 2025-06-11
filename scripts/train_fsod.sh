#!/usr/bin/env bash
# Simple training script for the FSOD dataset
# Usage: bash scripts/train_fsod.sh [SPLIT] [SHOT] [OUTPUT_ROOT] [BASE_ITERS] [FINE_ITERS]
# BASE_ITERS and FINE_ITERS optionally override SOLVER.MAX_ITER for the two stages

SPLIT=${1:-1}
SHOT=${2:-10}
OUTPUT_ROOT=${3:-./checkpoints/fsod}
BASE_ITERS=${4:-0}
FINE_ITERS=${5:-0}

FINE_EXTRA=""
if [ "$FINE_ITERS" -gt 0 ]; then
    FINE_EXTRA="SOLVER.MAX_ITER ${FINE_ITERS}"
fi

# Stage 1: train the base detector
BASE_EXTRA=""
if [ "$BASE_ITERS" -gt 0 ]; then
    BASE_EXTRA="SOLVER.MAX_ITER ${BASE_ITERS}"
fi

python tools/train_net.py --num-gpus 3 \
    --config-file configs/RFS/base-training/R101_FPN_base_training_split${SPLIT}.yml \
    --opts OUTPUT_DIR ${OUTPUT_ROOT}/faster_rcnn_R_101_FPN_base${SPLIT} ${BASE_EXTRA}

# Initialize novel-class weights
python tools/ckpt_surgery.py \
    --src1 ${OUTPUT_ROOT}/faster_rcnn_R_101_FPN_base${SPLIT}/model_final.pth \
    --method randinit \
    --save-dir ${OUTPUT_ROOT}/faster_rcnn_R_101_FPN_all${SPLIT}

# Stage 2: fine-tune on novel data
python tools/train_net.py --num-gpus 3 \
    --config-file configs/RFS/split${SPLIT}/${SHOT}shot_GPB_PFB_proloss.yml \
    --opts MODEL.WEIGHTS ${OUTPUT_ROOT}/faster_rcnn_R_101_FPN_all${SPLIT}/model_surgery.pth \
           OUTPUT_DIR ${OUTPUT_ROOT}/split${SPLIT}_${SHOT}shot_GPB_PFB_proloss ${FINE_EXTRA}

