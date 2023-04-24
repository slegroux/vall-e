#!/usr/bin/env bash
# export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
set -x

# small
# decoder_dim=128
# nhead=4
# num_decoder_layers=4
# max_duration=24
# filter_max_duration=14
# exp_dir=exp/valle_nano
# num_epochs=2000
# start_epoch=19

# FULL

# MODEL
decoder_dim=1024
nhead=16
num_decoder_layers=12

# PARAMS
base_lr=0.05
warmup_steps=200
average_period=0

# EPOCHS
num_epochs=250
start_epoch=1
start_batch=0
accumulate_grad_steps=4

# DATA FILTERING
max_duration=30
filter_max_duration=14
filter_min_duration=0.5

# PATHS
exp_dir=exp/valle
  
python3 bin/trainer.py \
    --num-buckets 6 --dtype "float32" --save-every-n 10000 \
    --num-epochs ${num_epochs} --start-epoch ${start_epoch} --start-batch ${start_batch} --accumulate-grad-steps ${accumulate_grad_steps} \
    --decoder-dim ${decoder_dim} --nhead ${nhead} --num-decoder-layers ${num_decoder_layers} --prefix-mode 1 \
    --model-name valle --share-embedding true --norm-first true --add-prenet false \
    --base-lr ${base_lr} --warmup-steps ${warmup_steps} --average-period ${average_period} \
    --max-duration ${max_duration} --filter-max-duration ${filter_max_duration} --filter-min-duration ${filter_min_duration} \
    --exp-dir ${exp_dir} \
    --world-size 8 
    # --drop-last true

# python bin/trainer.py --max-duration 30 --filter-min-duration 0.5 --filter-max-duration 14 \
#     --model-name "VALL-E" --norm-first true --add-prenet false --dtype "float32" \
#     --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
#     --base-lr 0.05 --warmup-steps 200 --average-period 0 --accumulate-grad-steps 1 \
#     --num-epochs 40 --start-epoch 1 --start-batch 0 \
#     --exp-dir exp/valle \
#     --world-size 8 \
#     --drop-last true