#!/bin/bash

python -m mlx_lm lora \
  --model mlx-community/Ministral-8B-Instruct-2410-4bit \
  --data data \
  --train \
  --fine-tune-type lora \
  --batch-size 4 \
  --num-layers 16 \
  --iters 100 \
  --adapter-path adapters
