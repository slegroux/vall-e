#!/usr/bin/env bash
# export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
set -x

# small
decoder_dim=128
nhead=4
num_decoder_layers=4
max_duration=24
filter_max_duration=14
exp_dir=exp/valle_nano
num_epochs=2000
start_epoch=19

# full
# decoder_dim=1024
# nhead=16
# num_decoder_layers=12
# exp_dir=exp/valle
  
python3 bin/trainer.py \
    --num-epochs ${num_epochs} --start-epoch ${start_epoch} \
    --decoder-dim ${decoder_dim} --nhead ${nhead} --num-decoder-layers ${num_decoder_layers} \
    --max-duration ${max_duration} --filter-max-duration ${filter_max_duration} \
    --model-name valle \
    --exp-dir ${exp_dir} \
    --world-size 8 

