"""
Text Preprocessing Module
=========================

Functions for cleaning text and tokenization for multilingual
sentiment analysis.
"""

import re
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Union, Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import load_config, ensure_dir, setup_logging

logger = logging.getLogger(__name__)


def clean_text(
    text: str,
    remove_urls: bool = True,
    remove_html: bool = True,
    normalize_whitespace: bool = True,
    preserve_emojis: bool = True,
    lowercase: bool = False
) -> str:
    """
    Clean and normalize text for sentiment analysis.
    
    Args:
        text: Input text string
        remove_urls: Whether to remove URLs
        remove_html: Whether to remove HTML tags
        normalize_whitespace: Whether to normalize whitespace
        preserve_emojis: Whether to keep emojis (useful for sentiment)
        lowercase: Whether to convert to lowercase
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    if remove_html:
        text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        text = re.sub(r'https?://\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove @mentions and #hashtags (keep the word)
    text = re.sub(r'[@#](\w+)', r'\1', text)
    
    # Normalize whitespace
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    # Lowercase (optional - multilingual models often work better without)
    if lowercase:
        text = text.lower()
    
    return text


def clean_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    **clean_kwargs
) -> pd.DataFrame:
    """
    Apply text cleaning to a DataFrame.
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column
        **clean_kwargs: Arguments passed to clean_text
        
    Returns:
        DataFrame with cleaned text
    """
    df = df.copy()
    
    # Clean text
    df[text_column] = df[text_column].apply(
        lambda x: clean_text(x, **clean_kwargs)
    )
    
    # Remove empty texts
    original_len = len(df)
    df = df[df[text_column].str.len() > 0]
    
    if len(df) < original_len:
        logger.info(f"Removed {original_len - len(df)} empty samples after cleaning")
    
    return df


def build_tokenizer(model_name: str = "xlm-roberta-base"):
    """
    Build and return a Hugging Face tokenizer.
    
    Args:
        model_name: Name of the pretrained model
        
    Returns:
        AutoTokenizer instance
    """
    from transformers import AutoTokenizer
    
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return tokenizer


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            labels: List of labels (0 or 1)
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def tokenize_and_cache(
    dataset_csv: Union[str, Path],
    tokenizer,
    max_length: int = 128,
    cache_path: Optional[Union[str, Path]] = None
) -> SentimentDataset:
    """
    Tokenize dataset and optionally cache to disk.
    
    Args:
        dataset_csv: Path to CSV file with 'text' and 'label' columns
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        cache_path: Optional path to save/load cached dataset
        
    Returns:
        SentimentDataset instance
    """
    # Check for cached version
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info(f"Loading cached dataset from {cache_path}")
            cached = torch.load(cache_path)
            return SentimentDataset(
                texts=cached["texts"],
                labels=cached["labels"],
                tokenizer=tokenizer,
                max_length=max_length
            )
    
    # Load and process data
    logger.info(f"Loading data from {dataset_csv}")
    df = pd.read_csv(dataset_csv)
    
    # Clean text
    df = clean_dataframe(df)
    
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    
    # Create dataset
    dataset = SentimentDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Cache if requested
    if cache_path:
        ensure_dir(cache_path.parent)
        torch.save({
            "texts": texts,
            "labels": labels
        }, cache_path)
        logger.info(f"Cached dataset to {cache_path}")
    
    return dataset


def create_hf_dataset(
    csv_path: Union[str, Path],
    tokenizer,
    max_length: int = 128
) -> "datasets.Dataset":
    """
    Create a Hugging Face Dataset from CSV for use with Trainer.
    
    Args:
        csv_path: Path to CSV file
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Hugging Face Dataset object
    """
    from datasets import Dataset as HFDataset
    
    df = pd.read_csv(csv_path)
    df = clean_dataframe(df)
    
    # Create HF dataset
    dataset = HFDataset.from_pandas(df)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Rename label column if needed
    if "label" in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    return tokenized_dataset


def load_and_tokenize_splits(
    data_dir: Union[str, Path],
    language: str,
    tokenizer,
    max_length: int = 128,
    cache_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Load and tokenize all splits for a language.
    
    Args:
        data_dir: Base data directory
        language: Language code
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        cache_dir: Optional cache directory
        
    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    data_dir = Path(data_dir)
    splits = {}
    
    for split in ["train", "val", "test"]:
        csv_path = data_dir / language / f"{split}.csv"
        
        if not csv_path.exists():
            logger.warning(f"Split not found: {csv_path}")
            continue
        
        cache_path = None
        if cache_dir:
            cache_path = Path(cache_dir) / language / f"{split}_tokenized.pt"
        
        logger.info(f"Processing {language}/{split}...")
        splits[split] = create_hf_dataset(csv_path, tokenizer, max_length)
        logger.info(f"  Loaded {len(splits[split])} samples")
    
    return splits


def get_text_statistics(texts: List[str], tokenizer) -> Dict[str, float]:
    """
    Compute text statistics after tokenization.
    
    Args:
        texts: List of text samples
        tokenizer: Hugging Face tokenizer
        
    Returns:
        Dictionary of statistics
    """
    lengths = []
    
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(tokens))
    
    import numpy as np
    lengths = np.array(lengths)
    
    return {
        "mean_tokens": float(np.mean(lengths)),
        "std_tokens": float(np.std(lengths)),
        "min_tokens": int(np.min(lengths)),
        "max_tokens": int(np.max(lengths)),
        "median_tokens": float(np.median(lengths)),
        "p95_tokens": float(np.percentile(lengths, 95)),
        "p99_tokens": float(np.percentile(lengths, 99))
    }


def main():
    """Main entry point for preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess and tokenize sentiment datasets"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed CSV files"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "bn", "all"],
        default="all",
        help="Language to preprocess"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for tokenizer (overrides config)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum sequence length (overrides config)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cached",
        help="Directory for cached tokenized data"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only compute statistics, don't tokenize"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load config
    config = load_config(args.config)
    
    model_name = args.model_name or config["model"]["default_model"]
    max_length = args.max_length or config["model"]["max_length"]
    
    # Build tokenizer
    tokenizer = build_tokenizer(model_name)
    
    # Determine languages to process
    languages = ["en", "bn"] if args.language == "all" else [args.language]
    
    for lang in languages:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {lang.upper()} data")
        logger.info(f"{'='*50}")
        
        data_dir = Path(args.data_dir)
        
        for split in ["train", "val", "test"]:
            csv_path = data_dir / lang / f"{split}.csv"
            
            if not csv_path.exists():
                logger.warning(f"File not found: {csv_path}")
                continue
            
            # Load data
            df = pd.read_csv(csv_path)
            logger.info(f"\n{split.upper()} split: {len(df)} samples")
            
            # Clean data
            df_clean = clean_dataframe(df)
            logger.info(f"  After cleaning: {len(df_clean)} samples")
            
            # Compute statistics
            stats = get_text_statistics(df_clean["text"].tolist(), tokenizer)
            logger.info(f"  Token statistics:")
            for key, value in stats.items():
                logger.info(f"    {key}: {value:.2f}")
            
            if not args.stats_only:
                # Create HF dataset
                hf_dataset = create_hf_dataset(csv_path, tokenizer, max_length)
                
                # Save cached version
                cache_path = Path(args.cache_dir) / lang / f"{split}_hf"
                ensure_dir(cache_path.parent)
                hf_dataset.save_to_disk(str(cache_path))
                logger.info(f"  Saved to {cache_path}")
    
    logger.info("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
