#!/usr/bin/env bash

# MODEL
decoder_dim=256
nhead=8
num_decoder_layers=6

# PARAMS
base_lr=0.05
warmup_steps=200
average_period=0

# EPOCHS
num_epochs=1000
start_epoch=240
start_batch=0
accumulate_grad_steps=1

# DATA FILTERING
max_duration=72
filter_max_duration=14
filter_min_duration=0.5

world_size=8
# exp_dir=exp/valle_D${decoder_dim}_H${nhead}_L${num_decoder_layers}_${world_size}GPU_LR${lr}_WU${warmup_steps}
exp_dir=exp/valle_Dim256H8L6_LR05


python bin/trainer.py --manifest-dir data/tokenized --text-tokens data/tokenized/unique_text_tokens.k2symbols \
      --max-duration ${max_duration} --filter-max-duration ${filter_max_duration} \
      --num-buckets 6 --dtype "float32" --save-every-n 10000 \
      --model-name valle --norm-first true --add-prenet false \
      --decoder-dim ${decoder_dim} --nhead ${nhead} --num-decoder-layers ${num_decoder_layers} --prefix-mode 0 \
      --base-lr ${base_lr} --warmup-steps ${warmup_steps} \
      --num-epochs ${num_epochs} --start-epoch ${start_epoch} --start-batch 0 --accumulate-grad-steps ${accumulate_grad_steps} \
      --exp-dir ${exp_dir} \
      --world-size ${world_size} \
      --drop-last true