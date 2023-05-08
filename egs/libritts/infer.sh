#!/usr/bin/env bash

# small
# decoder_dim=128
# nhead=4
# num_decoder_layers=4
# epoch=1989

# full
decoder_dim=1024
nhead=16
num_decoder_layers=12
prefix_mode=1
exp_dir=exp/valle_1gpu
epoch=97

python3 bin/infer.py --output-dir output \
    --cuda 0 \
    --model-name "VALL-E" --norm-first true --add-prenet false --share-embedding true \
    --decoder-dim ${decoder_dim} --nhead ${nhead} --num-decoder-layers ${num_decoder_layers} --prefix-mode ${prefix_mode} \
    --text-prompts "KNOT one point one five miles per hour." \
    --audio-prompts ./prompts/8463_294825_000043_000000.wav \
    --text "this is the multi-speaker version." \
    --checkpoint=${exp_dir}/best-valid-loss.pt
    # epoch-${epoch}.pt
    # --checkpoint=${exp_dir}/batch-17fc695a-07a0-ca6e-0822-e8f36c031199.pt
    # --checkpoint=exp/valle_nano/best-train-loss.pt

