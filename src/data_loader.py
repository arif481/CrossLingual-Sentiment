"""
Data Loader Module
==================

Functions for downloading, loading, and preparing sentiment datasets
for English and Bengali languages.
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .utils import load_config, ensure_dir, setup_logging

logger = logging.getLogger(__name__)


def download_english_dataset(
    output_dir: Union[str, Path],
    dataset_name: str = "sst2"
) -> pd.DataFrame:
    """
    Download English sentiment dataset (SST-2 from GLUE).
    
    Args:
        output_dir: Directory to save downloaded data
        dataset_name: Name of dataset ('sst2' or 'imdb')
        
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    from datasets import load_dataset
    
    output_dir = ensure_dir(Path(output_dir) / "en")
    logger.info(f"Downloading English dataset: {dataset_name}")
    
    if dataset_name == "sst2":
        # Load SST-2 from GLUE
        dataset = load_dataset("glue", "sst2")
        
        # Combine train and validation (SST-2 test has no labels)
        train_df = pd.DataFrame({
            "text": dataset["train"]["sentence"],
            "label": dataset["train"]["label"]
        })
        val_df = pd.DataFrame({
            "text": dataset["validation"]["sentence"],
            "label": dataset["validation"]["label"]
        })
        
        df = pd.concat([train_df, val_df], ignore_index=True)
        
    elif dataset_name == "imdb":
        # Load IMDB dataset
        dataset = load_dataset("imdb")
        
        train_df = pd.DataFrame({
            "text": dataset["train"]["text"],
            "label": dataset["train"]["label"]
        })
        test_df = pd.DataFrame({
            "text": dataset["test"]["text"],
            "label": dataset["test"]["label"]
        })
        
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        raise ValueError(f"Unknown English dataset: {dataset_name}")
    
    # Save raw data
    raw_path = output_dir / f"{dataset_name}_raw.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved English dataset to {raw_path} ({len(df)} samples)")
    
    return df


def download_bengali_dataset(
    output_dir: Union[str, Path],
    use_kaggle_fallback: bool = False
) -> pd.DataFrame:
    """
    Download Bengali sentiment dataset from Hugging Face or Kaggle.
    
    Args:
        output_dir: Directory to save downloaded data
        use_kaggle_fallback: If True, attempt Kaggle download if HF fails
        
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    from datasets import load_dataset
    
    output_dir = ensure_dir(Path(output_dir) / "bn")
    logger.info("Downloading Bengali sentiment dataset")
    
    try:
        # Try Hugging Face dataset first
        dataset = load_dataset("sepidmnorozy/Bengali_sentiment")
        
        # Extract text and labels
        if "train" in dataset:
            df = pd.DataFrame(dataset["train"])
        else:
            # Some datasets only have one split
            df = pd.DataFrame(dataset[list(dataset.keys())[0]])
        
        # Rename columns if needed
        column_mapping = {
            "sentence": "text",
            "review": "text", 
            "sentiment": "label",
            "polarity": "label"
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure we have the required columns
        if "text" not in df.columns:
            text_col = [c for c in df.columns if c not in ["label", "id"]][0]
            df = df.rename(columns={text_col: "text"})
        
        logger.info(f"Successfully loaded Bengali dataset from Hugging Face")
        
    except Exception as e:
        logger.warning(f"Failed to load from Hugging Face: {e}")
        
        if use_kaggle_fallback:
            df = _download_bengali_from_kaggle(output_dir)
        else:
            logger.info("Using demo Bengali data as fallback")
            # Use demo data as fallback
            demo_path = Path("data/raw/demo/demo_bn.csv")
            if demo_path.exists():
                df = pd.read_csv(demo_path)
            else:
                raise RuntimeError(
                    "Bengali dataset unavailable. Set up Kaggle API or use demo data.\n"
                    "See data/README.md for Kaggle setup instructions."
                )
    
    # Standardize labels
    df = standardize_labels(df)
    
    # Keep only required columns
    df = df[["text", "label"]]
    
    # Save raw data
    raw_path = output_dir / "bengali_sentiment_raw.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved Bengali dataset to {raw_path} ({len(df)} samples)")
    
    return df


def _download_bengali_from_kaggle(output_dir: Path) -> pd.DataFrame:
    """
    Download Bengali dataset from Kaggle (requires API setup).
    
    Args:
        output_dir: Directory to save downloaded data
        
    Returns:
        DataFrame with text and label columns
    """
    try:
        import kaggle
        
        logger.info("Attempting Kaggle download for Bengali sentiment data")
        
        # Download dataset
        kaggle.api.dataset_download_files(
            "cryptexcode/bangla-sentiment-analysis",
            path=str(output_dir),
            unzip=True
        )
        
        # Find and load the CSV
        csv_files = list(output_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in Kaggle download")
        
        df = pd.read_csv(csv_files[0])
        logger.info(f"Loaded Kaggle Bengali dataset: {len(df)} samples")
        
        return df
        
    except ImportError:
        raise RuntimeError(
            "Kaggle library not installed. Run: pip install kaggle\n"
            "Then set up API token: https://www.kaggle.com/docs/api"
        )
    except Exception as e:
        raise RuntimeError(f"Kaggle download failed: {e}")


def standardize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize labels to binary format {0: negative, 1: positive}.
    
    Args:
        df: DataFrame with 'label' column
        
    Returns:
        DataFrame with standardized labels
    """
    df = df.copy()
    
    if "label" not in df.columns:
        raise ValueError("DataFrame must have 'label' column")
    
    # Check current label values
    unique_labels = df["label"].unique()
    logger.info(f"Original unique labels: {unique_labels}")
    
    # Map various label formats to binary
    label_mapping = {}
    
    # String labels
    str_to_binary = {
        "positive": 1, "pos": 1, "1": 1, "good": 1,
        "negative": 0, "neg": 0, "0": 0, "bad": 0,
        "neutral": None  # Will be dropped
    }
    
    for label in unique_labels:
        if isinstance(label, str):
            label_lower = label.lower().strip()
            if label_lower in str_to_binary:
                label_mapping[label] = str_to_binary[label_lower]
            else:
                # Try to parse as int
                try:
                    label_mapping[label] = int(label)
                except ValueError:
                    logger.warning(f"Unknown string label: {label}")
        elif isinstance(label, (int, np.integer)):
            # Binary labels
            if label in [0, 1]:
                label_mapping[label] = label
            # Multi-class to binary (e.g., 0-4 scale)
            elif label in [0, 1, 2]:
                if label == 2:  # Neutral
                    label_mapping[label] = None
                else:
                    label_mapping[label] = label
            elif label in [3, 4, 5]:
                label_mapping[label] = 1  # Positive
            else:
                label_mapping[label] = 1 if label > 2 else 0
        elif isinstance(label, (float, np.floating)):
            label_mapping[label] = 1 if label > 0.5 else 0
    
    # Apply mapping
    df["label"] = df["label"].map(label_mapping)
    
    # Remove neutral/unknown labels
    original_len = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    
    if len(df) < original_len:
        logger.info(f"Removed {original_len - len(df)} samples with neutral/unknown labels")
    
    logger.info(f"Standardized labels. Distribution: {df['label'].value_counts().to_dict()}")
    
    return df


def split_and_save(
    df: pd.DataFrame,
    out_dir: Union[str, Path],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Dict[str, pd.DataFrame]:
    """
    Perform stratified split and save to CSV files.
    
    Args:
        df: Input DataFrame with 'text' and 'label' columns
        out_dir: Output directory for split files
        seed: Random seed for reproducibility
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set  
        test_ratio: Proportion for test set
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    out_dir = ensure_dir(out_dir)
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed,
        stratify=df["label"]
    )
    
    # Second split: val vs test
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_ratio,
        random_state=seed,
        stratify=temp_df["label"]
    )
    
    # Save splits
    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }
    
    for split_name, split_df in splits.items():
        path = out_dir / f"{split_name}.csv"
        split_df.to_csv(path, index=False)
        logger.info(f"Saved {split_name}: {len(split_df)} samples to {path}")
    
    # Print statistics
    for split_name, split_df in splits.items():
        pos_count = (split_df["label"] == 1).sum()
        neg_count = (split_df["label"] == 0).sum()
        logger.info(f"  {split_name}: {len(split_df)} total | "
                   f"pos: {pos_count} ({100*pos_count/len(split_df):.1f}%) | "
                   f"neg: {neg_count} ({100*neg_count/len(split_df):.1f}%)")
    
    return splits


def load_processed_data(
    data_dir: Union[str, Path],
    language: str,
    split: str = "train"
) -> pd.DataFrame:
    """
    Load processed data for a specific language and split.
    
    Args:
        data_dir: Base data directory
        language: Language code ('en' or 'bn')
        split: Data split ('train', 'val', or 'test')
        
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    path = Path(data_dir) / language / f"{split}.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    df = pd.read_csv(path)
    logger.info(f"Loaded {language}/{split}: {len(df)} samples")
    
    return df


def create_combined_dataset(
    data_dir: Union[str, Path],
    languages: List[str],
    split: str = "train",
    balance: bool = True
) -> pd.DataFrame:
    """
    Create combined dataset from multiple languages.
    
    Args:
        data_dir: Base data directory
        languages: List of language codes
        split: Data split to combine
        balance: If True, undersample to balance languages
        
    Returns:
        Combined DataFrame with language column
    """
    dfs = []
    
    for lang in languages:
        df = load_processed_data(data_dir, lang, split)
        df["language"] = lang
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    if balance:
        # Find minimum samples per language
        min_samples = min(len(df) for df in dfs)
        
        balanced_dfs = []
        for lang in languages:
            lang_df = combined[combined["language"] == lang]
            if len(lang_df) > min_samples:
                lang_df = lang_df.sample(n=min_samples, random_state=42)
            balanced_dfs.append(lang_df)
        
        combined = pd.concat(balanced_dfs, ignore_index=True)
        logger.info(f"Balanced combined dataset: {len(combined)} samples")
    
    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return combined


def get_dataset_stats(df: pd.DataFrame) -> Dict[str, any]:
    """
    Compute dataset statistics.
    
    Args:
        df: DataFrame with 'text' and 'label' columns
        
    Returns:
        Dictionary of statistics
    """
    text_lengths = df["text"].str.len()
    word_counts = df["text"].str.split().str.len()
    
    stats = {
        "total_samples": len(df),
        "positive_samples": int((df["label"] == 1).sum()),
        "negative_samples": int((df["label"] == 0).sum()),
        "positive_ratio": float((df["label"] == 1).mean()),
        "avg_text_length": float(text_lengths.mean()),
        "max_text_length": int(text_lengths.max()),
        "min_text_length": int(text_lengths.min()),
        "avg_word_count": float(word_counts.mean()),
        "max_word_count": int(word_counts.max()),
    }
    
    return stats


def main():
    """Main entry point for data loading."""
    parser = argparse.ArgumentParser(
        description="Download and prepare sentiment datasets"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "bn", "all"],
        default="all",
        help="Language to download"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for raw data"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--use-demo",
        action="store_true",
        help="Use demo data instead of downloading"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load config
    config = load_config(args.config)
    seed = config["training"]["seed"]
    split_config = config["data_split"]
    
    datasets = {}
    
    if args.use_demo:
        # Use demo data
        logger.info("Using demo datasets")
        
        demo_en = Path("data/raw/demo/demo_en.csv")
        demo_bn = Path("data/raw/demo/demo_bn.csv")
        
        if args.language in ["en", "all"] and demo_en.exists():
            datasets["en"] = pd.read_csv(demo_en)
            logger.info(f"Loaded demo English data: {len(datasets['en'])} samples")
            
        if args.language in ["bn", "all"] and demo_bn.exists():
            datasets["bn"] = pd.read_csv(demo_bn)
            logger.info(f"Loaded demo Bengali data: {len(datasets['bn'])} samples")
    else:
        # Download full datasets
        if args.language in ["en", "all"]:
            datasets["en"] = download_english_dataset(args.output_dir)
        
        if args.language in ["bn", "all"]:
            datasets["bn"] = download_bengali_dataset(args.output_dir)
    
    # Process and split each dataset
    for lang, df in datasets.items():
        logger.info(f"\nProcessing {lang} dataset...")
        
        # Standardize labels
        df = standardize_labels(df)
        
        # Print stats
        stats = get_dataset_stats(df)
        logger.info(f"Dataset statistics for {lang}:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Split and save
        out_dir = Path(args.processed_dir) / lang
        split_and_save(
            df,
            out_dir,
            seed=seed,
            train_ratio=split_config["train_ratio"],
            val_ratio=split_config["val_ratio"],
            test_ratio=split_config["test_ratio"]
        )
    
    logger.info("\nData loading complete!")


if __name__ == "__main__":
    main()
