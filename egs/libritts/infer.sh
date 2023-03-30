#!/usr/bin/env bash

decoder_dim=128
nhead=4
num_decoder_layers=4
epoch=1989

python3 bin/infer.py --output-dir infer/demos \
    --model-name valle --norm-first true --add-prenet false \
    --decoder-dim ${decoder_dim} --nhead ${nhead} --num-decoder-layers ${num_decoder_layers} \
    --text-prompts "KNOT one point one five miles per hour." \
    --audio-prompts ./prompts/8463_294825_000043_000000.wav \
    --text "To get up and running quickly just follow the steps below." \
    --checkpoint=exp/valle_nano/epoch-${epoch}.pt
    # --checkpoint=exp/valle_nano/best-train-loss.pt
