# LLM Fine-tuning Pipeline on Apple Silicon GPUs with MLX

This is an LLM fine-tuning pipeline for Apple Silicon GPUs using MLX. The project enables:

Data Collection & Preparation:

Web scraping with `scripts/web_scraper.py` to extract content from websites (like Wikipedia)
Data preprocessing with `scripts/prepare_jsonl_data.py` to convert scraped CSV data into JSONL training format
Model Training:

Fine-tuning the `mlx-community/Ministral-8B-Instruct-2410-4bit` model using LoRA (Low-Rank Adaptation)
Training script `scripts/train.sh` with configurable parameters (batch size, iterations, learning rate)
Testing capabilities via `scripts/test.sh`
Key Features:

Optimized for Apple Silicon using MLX framework
LoRA fine-tuning for efficient training with limited resources
Multiple data formats supported (Q&A, instruction-following, chat)
Automated pipeline from web scraping to model inference

Workflow:

1. Scrape web content → CSV
2. Convert CSV → JSONL training data
3. Fine-tune model with LoRA
4. Generate responses with the adapted model

The project is designed for creating domain-specific AI assistants by training on custom web content.

## Usage

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You need huggingface account and token to download the model.

```sh
hf auth login
hf download mlx-community/Ministral-8B-Instruct-2410-4bit
```

```sh
python ./scripts/web_scraper.py https://en.wikipedia.org/wiki/Yugoslavia -p 20 -o dataset/data.csv
```

```sh
python ./scripts/prepare_jsonl_data.py dataset/data.csv
```

```sh
./scripts/train.sh
```

```sh
./scripts/test.sh
```

Example inference:

```sh
./scripts/run.sh "Explain the history of the Balkans"
```

or

```sh
./scripts/run.sh "Who was the president of Yugoslavia?"
```

## Sample



https://github.com/user-attachments/assets/b2abe016-4e15-463c-a962-13f6a75350aa


