#!/usr/bin/env bash


python bin/trainer.py --manifest-dir data/fbank --text-tokens data/fbank/unique_text_tokens.k2symbols --max-duration 72 --filter-max-duration 14 \
      --num-buckets 6 --dtype "float32" --save-every-n 10000 \
      --model-name valle --norm-first true --add-prenet false \
      --decoder-dim 256 --nhead 8 --num-decoder-layers 6 --prefix-mode 0 \
      --base-lr 0.05 --warmup-steps 200 \
      --num-epochs 100 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 1 \
      --exp-dir exp/valle_Dim256H8L6_LR05