#!/usr/bin/env bash
set -x
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
# exp_dir=exp/valle_ar_nar_1gpu
# exp_dir=exp/max_dur-40_gpu-1
exp_dir=exp/libritts460-decoder_dim1024-nhead16-nlayers12-max_dur40-lr0.05-warmup200-world_size_8
epoch=463

# female
text_prompt="KNOT one point one five miles per hour."
audio_prompt=./prompts/8463_294825_000043_000000.wav
# male
text_prompt="This I read with great attention, while they sat silent."
audio_prompt=./prompts/8455_210777_000067_000000.wav
# obama
text_prompt="Nearly 10 years ago, america suffered the worst attack on our shores since pearl harbor."
audio_prompt=/data/en/zs_eval/obama.wav
# rishav
text_prompt="correct me if i'm wrong, like alex or alexia i think it's one of them will be picking this up from last year."
audio_prompt=/data/en/zs_eval/rishav.wav



python3 bin/infer.py --output-dir ${exp_dir}/output \
    --model-name "VALL-E" --norm-first true --add-prenet false --share-embedding true \
    --decoder-dim ${decoder_dim} --nhead ${nhead} --num-decoder-layers ${num_decoder_layers} --prefix-mode ${prefix_mode} \
    --text-prompts "${text_prompt}" \
    --audio-prompts ${audio_prompt} \
    --text "This is an example" \
    --checkpoint=${exp_dir}/best-valid-loss.pt
    # epoch-${epoch}.pt #best-valid-loss.pt #best-train-loss.pt
    # --checkpoint=${exp_dir}/batch-17fc695a-07a0-ca6e-0822-e8f36c031199.pt
    # --checkpoint=exp/valle_nano/best-train-loss.pt

