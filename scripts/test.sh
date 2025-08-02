#!/bin/bash

python -m mlx_lm lora \
  --model mlx-community/Ministral-8B-Instruct-2410-4bit \
  --data data \
  --adapter-path adapters \
  --test
