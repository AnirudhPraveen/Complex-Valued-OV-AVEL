#!/bin/bash
# OV-AVEL training-free baseline (frozen ImageBind, sqrt(relu(s_a * s_v)) fusion with
# the 'other' prompt), evaluated on close (seen) / open (unseen) / total.
#
#   OVAVEL_DATASET=ovavel|ave_ov  TEST_STRATEGY=v01|v0  GPU=0  bash run_baseline_v0.sh
set -euo pipefail
cd "$(dirname "$0")"
export OVAVEL_DATASET=${OVAVEL_DATASET:-ovavel}
GPU=${GPU:-0}
SEED=${SEED:-123}
TEST_STRATEGY=${TEST_STRATEGY:-v01}   # v01: with the 'other' class (used for AVE-OV in the paper)
BS=${BS:-1}
ROOT="./ExpResults/${OVAVEL_DATASET}/training_free_${TEST_STRATEGY}"
for SPLIT in close open total; do
    echo "[EVAL] training-free  split=${SPLIT}"
    CUDA_VISIBLE_DEVICES=${GPU} python baseline_v0_training_free.py \
        --model baseline_v0_training_free --seed ${SEED} --lr 0.001 --evaluate \
        --val_data_type ${SPLIT} --test_data_type ${SPLIT} --test_strategy_type ${TEST_STRATEGY} \
        --val_batch_size ${BS} --test_batch_size ${BS} --print_iter_freq 200 \
        --snapshot_pref "${ROOT}/eval_${SPLIT}"
done
bash ./scripts/print_results.sh "${ROOT}"
