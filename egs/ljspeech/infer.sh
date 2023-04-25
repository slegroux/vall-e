#!/usr/bin/env bash

decoder_dim=256
nhead=8
num_decoder_layers=6
prefix_mode=0
exp_dir=exp/valle_Dim256H8L6_LR05
epoch=240

python3 bin/infer.py --output-dir infer/demos \
    --model-name "VALL-E" --norm-first true --add-prenet false \
    --decoder-dim ${decoder_dim} --nhead ${nhead} --num-decoder-layers ${num_decoder_layers} --prefix-mode ${prefix_mode} \
    --text-prompts "KNOT one point one five miles per hour." \
    --audio-prompts ../libritts/prompts/8463_294825_000043_000000.wav \
    --text "To get up and running quickly just follow the steps below." \
    --checkpoint=${exp_dir}/checkpoint-30000.pt
    # --checkpoint=${exp_dir}/best-valid-loss.pt
    # --checkpoint=${exp_dir}/epoch-${epoch}.pt

