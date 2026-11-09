#!/bin/bash
# OPTIONAL: retrain the IF-phase -> mel CycleGAN from scratch.
# A trained generator already ships at checkpoints/cyclegan_if2mel.pt, so this
# is only needed to reproduce that checkpoint.
#
#   1) extract magnitude / phase / IF spectrograms from the dataset audio
#      (writes to $SPEC_DIR; needs well over 100 GB of disk)
#   2) train the unpaired CycleGAN (paper settings: 200 epochs, bs 16,
#      lr_G 2e-4, lr_D 1e-4, lambda_cyc 10, lambda_idt = lambda_dist = 2)
#
# Then point the main model at the new generator:
#   CYCLEGAN_CKPT=./checkpoints/cyclegan/cyclegan_best.pt bash run_complex_fusion.sh
set -euo pipefail
cd "$(dirname "$0")"

GPU=${GPU:-0}
SPEC_DIR=${SPEC_DIR:-./spectrograms}
export OVAVEL_DATASET=${OVAVEL_DATASET:-ovavel}

CUDA_VISIBLE_DEVICES=${GPU} python extract_spectrograms.py \
    --dataset "${OVAVEL_DATASET}" --output_dir "${SPEC_DIR}" --splits train val

CUDA_VISIBLE_DEVICES=${GPU} python cyclegan_phase2mel.py \
    --data_dir "${SPEC_DIR}" --checkpoint_dir ./checkpoints/cyclegan \
    --epochs 200 --batch_size 16 --lr_g 2e-4 --lr_d 1e-4 \
    --n_residual 9 --lambda_cycle 10 --lambda_identity 2 --lambda_dist 2
