#!/usr/bin/env bash

decoder_dim=256
nhead=8
num_decoder_layers=6
prefix_mode=0
exp_dir=exp/valle_Dim256H8L6_LR05
epoch=500
audio_input=/data/en/LJSpeech/LJSpeech-1.1/wavs/LJ001-0002.wav
transcript="in being comparatively modern."

python3 bin/infer.py --output-dir infer/demos \
    --model-name "VALL-E" --norm-first true --add-prenet false \
    --decoder-dim ${decoder_dim} --nhead ${nhead} --num-decoder-layers ${num_decoder_layers} --prefix-mode ${prefix_mode} \
    --text-prompts "${transcript}" \
    --audio-prompts ${audio_input} \
    --text "hello Kareem, how are you?" \
    --checkpoint=${exp_dir}/checkpoint-30000.pt
    # --checkpoint=${exp_dir}/best-train-loss.pt
    # --checkpoint=${exp_dir}/epoch-${epoch}.pt
    

    

