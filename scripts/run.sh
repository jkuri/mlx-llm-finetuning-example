#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"<your prompt here>\""
    echo "Example: $0 \"Who was the president of Yugoslavia?\""
    exit 1
fi

# Get the prompt from the first argument
PROMPT="$1"

python -m mlx_lm generate \
  --model mlx-community/Ministral-8B-Instruct-2410-4bit \
  --adapter-path adapters \
  --max-tokens 500 \
  --prompt "$PROMPT"
