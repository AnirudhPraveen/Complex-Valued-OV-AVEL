#!/bin/bash
# Train + evaluate one model, then report seen / unseen / total results.
#
# Knobs (environment variables):
#   OVAVEL_DATASET  ovavel (OV-AVEBench, default) | ave_ov (AVE-OV)
#   STREAMS         4 (RGB+iHSV, mel+CycleGAN-phase; default) | 2 (RGB, mel)
#   FUSION          complex (ComplexCNN, ours; default) | real (RealCNN ablation)
#                   | none (OV-AVEL fine-tuned baseline: sqrt(relu(s_a * s_v)), 2 streams)
#   PHASE_MODE      cyclegan (default) | imagebind (raw-IF ablation: learnable projector)
#   GPU SEED N_EPOCH LR TRAIN_BS TEST_BS
#   EXTRA_ARGS      extra flags for train_complex_fusion.py (e.g. "--debug" for a quick smoke test)
#
# Examples:
#   bash run_complex_fusion.sh                                   # ours, 4-stream, OV-AVEBench
#   STREAMS=2 bash run_complex_fusion.sh                         # ours, 2-stream
#   FUSION=none bash run_complex_fusion.sh                       # fine-tuned baseline
#   OVAVEL_DATASET=ave_ov bash run_complex_fusion.sh             # AVE-OV (10 epochs)
#
# The best epoch (avg of Acc / Seg-F1 / Eve-F1 on the 'total' split) is saved to
#   ExpResults/<dataset>/<run>/train/task_FullySupervised_best_model.pth.tar
# and evaluated on close (seen), open (unseen) and total.
set -euo pipefail
cd "$(dirname "$0")"

export OVAVEL_DATASET=${OVAVEL_DATASET:-ovavel}
STREAMS=${STREAMS:-4}
FUSION=${FUSION:-complex}
PHASE_MODE=${PHASE_MODE:-cyclegan}
GPU=${GPU:-0}
SEED=${SEED:-123}
LR=${LR:-5e-5}
TRAIN_BS=${TRAIN_BS:-32}
TEST_BS=${TEST_BS:-32}
if [[ "${OVAVEL_DATASET}" == "ave_ov" ]]; then N_EPOCH=${N_EPOCH:-10}; else N_EPOCH=${N_EPOCH:-5}; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ./scripts/model_config.sh   # sets CFG_OVERRIDES_JSON and RUN_NAME from the knobs above
export CFG_OVERRIDES_JSON

ROOT="./ExpResults/${OVAVEL_DATASET}/${RUN_NAME}_seed${SEED}"
CKPT="${ROOT}/train/task_FullySupervised_best_model.pth.tar"

echo "[TRAIN] ${RUN_NAME}  dataset=${OVAVEL_DATASET}  seed=${SEED}  epochs=${N_EPOCH}  -> ${ROOT}"
CUDA_VISIBLE_DEVICES=${GPU} python train_complex_fusion.py \
  --model "${RUN_NAME}" --seed ${SEED} --lr ${LR} --n_epoch ${N_EPOCH} \
  --train_batch_size ${TRAIN_BS} --test_batch_size ${TEST_BS} \
  --test_data_type total --test_strategy_type v1 \
  --eval_freq 1 --print_iter_freq 100 --clip_gradient 0.8 \
  --snapshot_pref "${ROOT}/train" ${EXTRA_ARGS:-}

[[ -f "${CKPT}" ]] || { echo "ERROR: best checkpoint not found at ${CKPT}" >&2; exit 1; }

for SPLIT in close open total; do
    echo "[EVAL] ${SPLIT}"
    CUDA_VISIBLE_DEVICES=${GPU} python train_complex_fusion.py \
      --model "${RUN_NAME}" --seed ${SEED} --evaluate --resume "${CKPT}" \
      --test_batch_size ${TEST_BS} --test_data_type ${SPLIT} --test_strategy_type v1 \
      --print_iter_freq 100 --snapshot_pref "${ROOT}/eval_${SPLIT}" ${EXTRA_ARGS:-}
done

bash ./scripts/print_results.sh "${ROOT}"
