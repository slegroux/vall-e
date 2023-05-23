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
num_epochs=2500
start_epoch=1
start_batch=0
accumulate_grad_steps=4

# DATA FILTERING
max_duration=32
filter_max_duration=14
filter_min_duration=0.5

# PATHS
exp_dir=exp/valle_${world_size}gpu

# ALL
train_stage=0

# AR
train_stage=1
max_duration=80
num_epochs=500
exp_dir=exp/valle_ar_nar_${world_size}gpu

# NAR
train_stage=0
max_duration=40
num_epochs=1000
world_size=8
# exp_dir=exp/valle_ar_nar_${world_size}gpu
name=libritts460
exp_dir=exp/${name}-decoder_dim${decoder_dim}-nhead${nhead}-nlayers${num_decoder_layers}-max_dur${max_duration}-lr${base_lr}-warmup${warmup_steps}-world_size_${world_size}
start_epoch=1


python3 bin/trainer.py \
    --train-stage ${train_stage} \
    --max-duration ${max_duration} --filter-max-duration ${filter_max_duration} --filter-min-duration ${filter_min_duration} \
    --num-buckets 6 --dtype "float32" --save-every-n 10000 --valid-interval 10000 \
    --model-name valle --share-embedding true --norm-first true --add-prenet false \
    --decoder-dim ${decoder_dim} --nhead ${nhead} --num-decoder-layers ${num_decoder_layers} --prefix-mode 1 \
    --base-lr ${base_lr} --warmup-steps ${warmup_steps} --average-period ${average_period} \
    --num-epochs ${num_epochs} --start-epoch ${start_epoch} --start-batch ${start_batch} --accumulate-grad-steps ${accumulate_grad_steps} \
    --exp-dir ${exp_dir} \
    --world-size ${world_size}
    # --drop-last true
