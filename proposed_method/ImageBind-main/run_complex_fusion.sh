#!/bin/bash
# Train + evaluate the 4-stream complex-valued fusion model on OV-AVEBench.
#   streams: RGB (real) + iHSV (imag) for vision, mel (real) + CycleGAN-translated
#            IF-phase (imag) for audio, fused by a ComplexCNN.
#
# The best checkpoint (selected on val=total) is evaluated on the close, open
# and total test splits, and a results table is printed at the end.
#
# Usage (from anywhere):
#   bash run_complex_fusion.sh
#   GPU=1 SEED=456 N_EPOCH=5 bash run_complex_fusion.sh
#   FUSION=real bash run_complex_fusion.sh        # 4-stream real-valued CNN ablation
set -euo pipefail
cd "$(dirname "$0")"

GPU=${GPU:-0}
SEED=${SEED:-123}
N_EPOCH=${N_EPOCH:-5}
LR=${LR:-5e-5}
TRAIN_BS=${TRAIN_BS:-32}
VAL_BS=${VAL_BS:-32}
TEST_BS=${TEST_BS:-32}
TEST_STRATEGY=${TEST_STRATEGY:-v1}
FUSION=${FUSION:-complex}                   # 'complex' (ours) or 'real' (ablation)
EXT=${EXT:-ComplexFusion_4Stream_${FUSION}}
export OVAVEL_DATASET=${OVAVEL_DATASET:-ovavel}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_NAME='complex_fusion'
SCRIPT='train_complex_fusion.py'
CYCLEGAN_CKPT=${CYCLEGAN_CKPT:-./checkpoints/cyclegan_if2mel.pt}
export CYCLEGAN_CKPT

if [[ ! -f "${CYCLEGAN_CKPT}" ]]; then
    echo "ERROR: CycleGAN generator not found at ${CYCLEGAN_CKPT}." >&2
    echo "       It ships with the repo; re-clone, or train one with run_train_cyclegan.sh." >&2
    exit 2
fi

CFG_OVERRIDES_JSON="$(sed "s/\"FUSION_CNN_TYPE\": \"complex\"/\"FUSION_CNN_TYPE\": \"${FUSION}\"/" configs/complex_fusion_4stream.json)"
export CFG_OVERRIDES_JSON

TRAIN_DIR="./ExpResults/${MODEL_NAME}/train_seed${SEED}_bs${TRAIN_BS}_lr${LR}/${EXT}"
CKPT="${TRAIN_DIR}/task_FullySupervised_best_model.pth.tar"

echo "[TRAIN] dataset=${OVAVEL_DATASET} fusion=${FUSION} seed=${SEED} epochs=${N_EPOCH} -> ${TRAIN_DIR}"
CUDA_VISIBLE_DEVICES=${GPU} python ${SCRIPT} \
  --lr ${LR} --seed ${SEED} --model ${MODEL_NAME} \
  --snapshot_pref "${TRAIN_DIR}" \
  --n_epoch ${N_EPOCH} \
  --train_batch_size ${TRAIN_BS} --val_batch_size ${VAL_BS} --test_batch_size ${TEST_BS} \
  --val_data_type total --test_data_type total \
  --test_strategy_type ${TEST_STRATEGY} \
  --eval_freq 1 --print_iter_freq 100 --clip_gradient 0.8

[[ -f "${CKPT}" ]] || { echo "ERROR: best checkpoint not found at ${CKPT}" >&2; exit 1; }

for SPLIT in close open total; do
    OUT_DIR="./ExpResults/${MODEL_NAME}/test_seed${SEED}_bs${TRAIN_BS}_lr${LR}/${EXT}__${SPLIT}"
    echo "[EVAL] split=${SPLIT} -> ${OUT_DIR}"
    CUDA_VISIBLE_DEVICES=${GPU} python ${SCRIPT} \
      --lr ${LR} --seed ${SEED} --model ${MODEL_NAME} --evaluate \
      --snapshot_pref "${OUT_DIR}" --resume "${CKPT}" \
      --val_batch_size ${TEST_BS} --test_batch_size ${TEST_BS} \
      --val_data_type total --test_data_type ${SPLIT} \
      --test_strategy_type ${TEST_STRATEGY} --print_iter_freq 100
done

echo
printf "%-6s | %-9s | %-10s | %-8s\n" "Split" "Accuracy" "Segment-F1" "Event-F1"
for SPLIT in close open total; do
    OUT_DIR="./ExpResults/${MODEL_NAME}/test_seed${SEED}_bs${TRAIN_BS}_lr${LR}/${EXT}__${SPLIT}"
    LOG=$(ls -t "${OUT_DIR}"/*-Eval.log 2>/dev/null | head -n 1 || true)
    LINE=$( [[ -n "${LOG}" ]] && grep "Evaluation results (acc)" "${LOG}" | tail -n 1 || true)
    ACC=$(echo "${LINE}" | sed -nE 's/.*\(acc\):[[:space:]]*([0-9.]+).*/\1/p')
    SEG=$(echo "${LINE}" | sed -nE 's/.*Segment-level F1score\):[[:space:]]*([0-9.]+).*/\1/p')
    EVE=$(echo "${LINE}" | sed -nE 's/.*Event-level F1score\):[[:space:]]*([0-9.]+).*/\1/p')
    printf "%-6s | %-9s | %-10s | %-8s\n" "${SPLIT}" "${ACC:-N/A}" "${SEG:-N/A}" "${EVE:-N/A}"
done
