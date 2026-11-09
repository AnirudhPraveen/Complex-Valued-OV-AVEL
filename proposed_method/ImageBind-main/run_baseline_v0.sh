#!/bin/bash

# =============================================================================
# Baseline V0: Training-free baseline
# =============================================================================
# Modes:
#   - v0:  Training-free (without 'other' category)
#   - v01: Training-free (with 'other' category)
# =============================================================================

SEED=123
MODEL_NAME='baseline_v0_training_free'
BS=4                         # batch size (same as original)
test_data_type='open'        # 'total', 'open', 'close'
test_strategy_type='v0'      # 'v0', 'v01' (training-free)

python baseline_v0_training_free.py \
    --lr 0.001 \
    --seed ${SEED} \
    --model ${MODEL_NAME} \
    --snapshot_pref "./ExpResults/${MODEL_NAME}/seed${SEED}_bs${BS}_dataType_${test_data_type}_strategy_${test_strategy_type}" \
    --train_batch_size ${BS} \
    --test_batch_size ${BS} \
    --n_epoch 2 \
    --eval_freq 1 \
    --print_iter_freq 100 \
    --test_data_type ${test_data_type} \
    --test_strategy_type ${test_strategy_type} \
    --evaluate
