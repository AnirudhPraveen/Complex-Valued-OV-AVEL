SEED=123
#SEED=456
MODEL_NAME='fine_tune_baseline_v1_train_fully'
N_EPOCH=2
LR=5e-5
TRAIN_BS=32 #!
VAL_BS=32   #! some val. data may not be used, however this will not significantly affect the model selection
TEST_BS=32
val_data_type='total'
test_data_type='total'   # 'close' for seen categories only, 'total' for all, 'open' for unseen only
test_strategy_type='v1' # 'v1' or 'v0' or 'v01', best: v1

#! Train
EXT="CycleGAN_Phase_iHSV_ComplexCNN_Open" #! CycleGAN IF→mel phase + iHSV + ComplexCNN fusion, test on open categories
python fine_tune_baseline_v1_train_fully_av.py \
  --lr ${LR} \
  --seed ${SEED} \
  --model ${MODEL_NAME} \
  --snapshot_pref "./ExpResults/${MODEL_NAME}/train_mode_seed${SEED}_bs${TRAIN_BS}_Lr${LR}_evalStrategy_${test_strategy_type}/${EXT}" \
  --n_epoch ${N_EPOCH} \
  --train_batch_size ${TRAIN_BS} \
  --val_batch_size ${VAL_BS} \
  --test_batch_size ${TEST_BS} \
  --val_data_type ${val_data_type} \
  --test_data_type ${test_data_type} \
  --test_strategy_type ${test_strategy_type} \
  --eval_freq 1 \
  --print_iter_freq 100 \
  --clip_gradient 0.8

# # # #! Test/Inference
# EXT="woSA_type_evenLast_K1_wTAOnlyselfLayer1_woTextTune_sqrtAV" #! (optimal, GEN1 w min-max norm) should carefully modify this to match the config.py
# CUDA_VISIBLE_DEVICES=0 python fine_tune_baseline_v1_train_fully.py \
#   --lr ${LR} \
#   --seed ${SEED} \
#   --model ${MODEL_NAME} \
#   --evaluate \
#   --snapshot_pref "./ExpResults/${MODEL_NAME}/test_mode_seed${SEED}_bs${TRAIN_BS}_Lr${LR}_evalStrategy_${test_strategy_type}/${EXT}" \
#   --resume "./ExpResults/fine_tune_baseline_v1_train_fully/train_mode_seed421_bs32_Lr5e-5_evalStrategy_v1/woSA_type_evenLast_K1_wCorrectTAOnlyselfLayer1_woTextTune_sqrtAV/task_FullySupervised_best_model.pth.tar" \
#   --test_batch_size ${TEST_BS} \
#   --test_data_type ${test_data_type} \
#   --test_strategy_type ${test_strategy_type} \
#   --print_iter_freq 100 \
#   --clip_gradient 0.8
