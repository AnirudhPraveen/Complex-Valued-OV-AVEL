#!/bin/bash
# Cross-dataset transfer: evaluate a checkpoint trained on one dataset on the
# UNSEEN (open) categories of the other dataset's test split. Text prompts are
# re-encoded for the target categories; the class-agnostic fusion CNN needs no change.
#
#   bash run_cross_dataset.sh <CKPT> <TARGET_DATASET: ovavel|ave_ov>
#   (set STREAMS / FUSION / PHASE_MODE to match the checkpoint's model)
#
# Example (OV-AVEBench -> AVE-OV, our 4-stream model):
#   bash run_cross_dataset.sh ExpResults/ovavel/4stream_complex_seed123/train/task_FullySupervised_best_model.pth.tar ave_ov
set -euo pipefail
cd "$(dirname "$0")"
CKPT=$1
export OVAVEL_DATASET=$2
GPU=${GPU:-0}
SEED=${SEED:-123}
TEST_BS=${TEST_BS:-32}
[[ -f "${CKPT}" ]] || { echo "ERROR: checkpoint not found: ${CKPT}" >&2; exit 1; }
source ./scripts/model_config.sh
export CFG_OVERRIDES_JSON
ROOT="./ExpResults/cross_dataset/to_${OVAVEL_DATASET}/${RUN_NAME}_seed${SEED}"
CUDA_VISIBLE_DEVICES=${GPU} python train_complex_fusion.py \
  --model "${RUN_NAME}" --seed ${SEED} --evaluate --resume "${CKPT}" \
  --test_batch_size ${TEST_BS} --test_data_type open --test_strategy_type v1 \
  --print_iter_freq 100 --snapshot_pref "${ROOT}/eval_open" ${EXTRA_ARGS:-}
bash ./scripts/print_results.sh "${ROOT}"
