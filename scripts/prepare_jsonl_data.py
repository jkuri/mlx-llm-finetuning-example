#!/usr/bin/env python3
"""
JSONL Data Preparation Script
Converts scraped CSV data into JSONL format for AI training
"""

import pandas as pd
import json
import random
import argparse
import os
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JSONLDataPreparer:
    def __init__(self, csv_file: str, output_dir: str = "data"):
        """
        Initialize the JSONL data preparer

        Args:
            csv_file: Path to the CSV file with scraped data
            output_dir: Directory to save JSONL files
        """
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.df = None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load CSV data and display basic info"""
        try:
            self.df = pd.read_csv(self.csv_file)
            logger.info(f"Loaded {len(self.df)} records from {self.csv_file}")
            logger.info(f"Columns: {list(self.df.columns)}")
            print("\nDataset preview:")
            print(self.df.head())
            return self.df
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise

    def create_qa_format(self, row: pd.Series) -> Dict[str, str]:
        """
        Create Q&A format training data from scraped content

        Args:
            row: DataFrame row with scraped data

        Returns:
            Dictionary with formatted training text
        """
        # Handle both old and new CSV formats
        page_title = row.get('page_title', row.get('title', '')).strip()
        section_title = row.get('section_title', '').strip()
        content = row.get('section_content', row.get('full_text', '')).strip()
        content_type = row.get('content_type', 'content')
        url = row.get('url', '')

        # Create context-aware prompts based on content type
        if content_type == 'section' and section_title:
            prompt_templates = [
                f"You are a helpful assistant. Based on this section about '{section_title}' from '{page_title}': {content[:800]}... Question: What does this section explain? Answer: This section explains {section_title.lower()} in the context of {page_title.lower()}.",

                f"Summarize this section titled '{section_title}': {content[:600]}... Summary: This section covers {section_title.lower()} and discusses the key points presented.",

                f"You are an expert on {page_title}. Content from section '{section_title}': {content[:700]}... Question: What are the main points? Answer: The main points about {section_title.lower()} include the information presented in this section."
            ]
        elif content_type == 'paragraph':
            prompt_templates = [
                f"You are a helpful assistant. Here's a paragraph from '{page_title}': {content[:800]}... Question: What is this paragraph about? Answer: This paragraph discusses aspects of {page_title.lower()}.",

                f"Explain this content from '{page_title}': {content[:600]}... Explanation: This content explains important information about {page_title.lower()}.",

                f"Based on this text about {page_title}: {content[:700]}... Question: What can you tell me about this? Answer: This text provides information about {page_title.lower()}."
            ]
        elif content_type == 'list':
            prompt_templates = [
                f"You are a helpful assistant. Here's a list from '{page_title}' under '{section_title}': {content[:600]}... Question: What does this list contain? Answer: This list contains key points about {section_title.lower() if section_title else page_title.lower()}.",

                f"Summarize this list about {page_title}: {content[:500]}... Summary: This list outlines important aspects of {page_title.lower()}.",

                f"Based on this list from {page_title}: {content[:700]}... Question: What are the main items? Answer: The main items relate to {page_title.lower()}."
            ]
        else:
            # Fallback for any other content
            prompt_templates = [
                f"You are a helpful assistant. Content about '{page_title}': {content[:800]}... Question: What is this about? Answer: This content discusses {page_title.lower()}.",

                f"Summarize this content: {content[:600]}... Summary: This content covers information about {page_title.lower() if page_title else 'the topic discussed'}."
            ]

        # Choose a random template
        prompt = random.choice(prompt_templates)

        return {"text": prompt, "source_url": url, "content_type": content_type}

    def create_instruction_format(self, row: pd.Series) -> Dict[str, str]:
        """
        Create instruction-following format for training

        Args:
            row: DataFrame row with scraped data

        Returns:
            Dictionary with instruction format
        """
        title = row.get('title', '').strip()
        content = row.get('full_text', '').strip()
        headings = row.get('headings', '').strip()
        url = row.get('url', '')

        # Create instruction-response pairs
        instructions = [
            {
                "instruction": "Summarize the main points of this web content.",
                "input": content[:1000],
                "output": f"The content titled '{title}' covers the following main points: {headings.replace(' | ', ', ') if headings else 'key information as presented'}."
            },
            {
                "instruction": "What is the title and main topic of this content?",
                "input": content[:800],
                "output": f"The title is '{title}' and the main topic focuses on the subject matter presented in the content."
            },
            {
                "instruction": "Extract the key headings and structure from this text.",
                "input": content[:600],
                "output": f"The key headings include: {headings if headings else 'main sections as organized in the content'}."
            }
        ]

        # Choose a random instruction format
        chosen = random.choice(instructions)

        return {
            "instruction": chosen["instruction"],
            "input": chosen["input"],
            "output": chosen["output"],
            "source_url": url
        }

    def create_chat_format(self, row: pd.Series) -> Dict[str, Any]:
        """
        Create chat/conversation format for training

        Args:
            row: DataFrame row with scraped data

        Returns:
            Dictionary with chat format
        """
        title = row.get('title', '').strip()
        content = row.get('full_text', '').strip()
        url = row.get('url', '')

        conversations = [
            {
                "messages": [
                    {"role": "user", "content": f"Can you tell me about this content: {content[:500]}..."},
                    {"role": "assistant", "content": f"This content is about '{title}' and discusses the topics presented in the text."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What's the main idea of this text?"},
                    {"role": "assistant", "content": f"The main idea revolves around '{title}' as described in the content."}
                ]
            }
        ]

        chosen = random.choice(conversations)
        chosen["source_url"] = url

        return chosen

    def prepare_data(self, format_type: str = "qa", train_ratio: float = 0.7,
                    test_ratio: float = 0.2, valid_ratio: float = 0.1) -> None:
        """
        Prepare and split data into train/test/validation sets

        Args:
            format_type: Type of format ('qa', 'instruction', 'chat')
            train_ratio: Ratio for training data
            test_ratio: Ratio for test data
            valid_ratio: Ratio for validation data
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Validate ratios
        if abs(train_ratio + test_ratio + valid_ratio - 1.0) > 0.001:
            raise ValueError("Train, test, and validation ratios must sum to 1.0")

        print(f"\nOriginal dataset info:")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")

        # Check for required columns (support both old and new formats)
        content_col = 'section_content' if 'section_content' in self.df.columns else 'full_text'
        title_col = 'page_title' if 'page_title' in self.df.columns else 'title'

        required_cols = [content_col, title_col]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"CSV must contain columns: {required_cols}")

        # More lenient filtering - check what we actually have
        print(f"\nData quality check:")
        print(f"Rows with non-null {content_col}: {self.df[content_col].notna().sum()}")
        print(f"Rows with non-empty {content_col}: {(self.df[content_col].str.len() > 0).sum()}")
        print(f"Rows with {content_col} > 50 chars: {(self.df[content_col].str.len() > 50).sum()}")
        print(f"Rows with non-null {title_col}: {self.df[title_col].notna().sum()}")
        print(f"Rows with non-empty {title_col}: {(self.df[title_col].str.len() > 0).sum()}")

        if 'content_type' in self.df.columns:
            print(f"Content type distribution: {self.df['content_type'].value_counts().to_dict()}")

        # Use more lenient filtering
        filtered_df = self.df[
            (self.df[content_col].notna()) &
            (self.df[content_col].str.len() > 30) &  # Reduced from 50 to 30 for sections
            (self.df[title_col].notna()) &
            (self.df[title_col].str.len() > 0)
        ].copy()

        logger.info(f"Filtered dataset: {len(filtered_df)} records (from {len(self.df)})")

        if len(filtered_df) == 0:
            logger.error("No records passed filtering! Check your data quality.")
            print("\nSample of original data:")
            print(self.df[['title', 'full_text']].head())
            raise ValueError("No valid records found after filtering")

        # Create formatted data based on type
        formatted_data = []
        failed_count = 0
        for idx, row in filtered_df.iterrows():
            try:
                if format_type == "qa":
                    formatted_data.append(self.create_qa_format(row))
                elif format_type == "instruction":
                    formatted_data.append(self.create_instruction_format(row))
                elif format_type == "chat":
                    formatted_data.append(self.create_chat_format(row))
                else:
                    raise ValueError(f"Unknown format type: {format_type}")
            except Exception as e:
                failed_count += 1
                logger.warning(f"Error processing row {idx}: {e}")
                continue

        logger.info(f"Successfully formatted {len(formatted_data)} records, {failed_count} failed")

        if len(formatted_data) == 0:
            raise ValueError("No records were successfully formatted!")

        # Shuffle the data
        random.shuffle(formatted_data)

        # Calculate split sizes - ensure we have at least 1 record in each split
        total_records = len(formatted_data)

        if total_records < 3:
            logger.warning(f"Only {total_records} records available. Putting all in training set.")
            train_data = formatted_data
            test_data = []
            valid_data = []
        else:
            train_split = max(1, int(total_records * train_ratio))
            test_split = max(1, int(total_records * test_ratio))

            # Ensure we don't exceed total records
            if train_split + test_split >= total_records:
                train_split = total_records - 2
                test_split = 1

            train_data = formatted_data[:train_split]
            test_data = formatted_data[train_split:train_split + test_split]
            valid_data = formatted_data[train_split + test_split:]

        logger.info(f"Final data splits - Train: {len(train_data)}, Test: {len(test_data)}, Valid: {len(valid_data)}")

        # Save to JSONL files
        self.save_jsonl(train_data, f"{self.output_dir}/train.jsonl")
        self.save_jsonl(test_data, f"{self.output_dir}/test.jsonl")
        self.save_jsonl(valid_data, f"{self.output_dir}/valid.jsonl")

        logger.info(f"Successfully created JSONL files in {self.output_dir}/")
        logger.info(f"Train: {len(train_data)} samples")
        logger.info(f"Test: {len(test_data)} samples")
        logger.info(f"Valid: {len(valid_data)} samples")

    def save_jsonl(self, data: List[Dict], filename: str) -> None:
        """Save data to JSONL format"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(data)} records to {filename}")
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Prepare JSONL data from scraped CSV')
    parser.add_argument('csv_file', help='Path to the CSV file with scraped data')
    parser.add_argument('-o', '--output', default='data',
                       help='Output directory for JSONL files (default: data)')
    parser.add_argument('-f', '--format', choices=['qa', 'instruction', 'chat'],
                       default='qa', help='Format type for training data (default: qa)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training data ratio (default: 0.7)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Test data ratio (default: 0.2)')
    parser.add_argument('--valid-ratio', type=float, default=0.1,
                       help='Validation data ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Initialize preparer
    preparer = JSONLDataPreparer(args.csv_file, args.output)

    try:
        # Load and prepare data
        preparer.load_data()
        preparer.prepare_data(
            format_type=args.format,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            valid_ratio=args.valid_ratio
        )

        print(f"\n✅ Successfully prepared JSONL data in '{args.output}' directory")
        print(f"Format: {args.format}")
        print(f"Files created: train.jsonl, test.jsonl, valid.jsonl")

    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
