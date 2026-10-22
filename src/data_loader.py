"""
Data Loader Module
==================

Functions for downloading, loading, and preparing sentiment datasets
for English (SST-2) and Bengali (SentNoB).
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .utils import load_config, ensure_dir, setup_logging

logger = logging.getLogger(__name__)


def map_labels(df: pd.DataFrame, label_map: Dict[int, int]) -> pd.DataFrame:
    """
    Map dataset-specific labels to {0: negative, 1: positive}.

    Labels missing from label_map (for example SentNoB's neutral class)
    are dropped.
    """
    df = df.copy()
    df["label"] = df["label"].map({int(k): int(v) for k, v in label_map.items()})
    dropped = int(df["label"].isna().sum())
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    if dropped:
        logger.info(f"Dropped {dropped} samples whose label is not in the label map")
    return df


def load_english_splits(config: Dict, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Load SST-2 as train/val/test.

    SST-2's test split has no labels, so the official validation split is
    used as the test set and a stratified part of train is held out for
    validation.
    """
    from datasets import load_dataset

    ds_config = config["datasets"]["english"]
    logger.info(f"Loading {ds_config['hf_dataset_id']} ({ds_config['hf_subset']})")
    dataset = load_dataset(ds_config["hf_dataset_id"], ds_config["hf_subset"])

    def to_frame(split):
        return pd.DataFrame({
            "text": dataset[split][ds_config["text_column"]],
            "label": dataset[split][ds_config["label_column"]],
        })

    full_train = to_frame("train")
    train_df, val_df = train_test_split(
        full_train,
        test_size=ds_config["val_size"],
        random_state=seed,
        stratify=full_train["label"],
    )
    return {"train": train_df, "val": val_df, "test": to_frame("validation")}


def load_bengali_splits(config: Dict) -> Dict[str, pd.DataFrame]:
    """
    Load SentNoB's train/val/test files and keep positive and negative comments.
    """
    from huggingface_hub import hf_hub_download

    ds_config = config["datasets"]["bengali"]
    splits = {}
    for split, filename in ds_config["files"].items():
        path = hf_hub_download(
            repo_id=ds_config["hf_dataset_id"],
            filename=filename,
            repo_type="dataset",
        )
        raw = pd.read_csv(path)
        df = pd.DataFrame({
            "text": raw[ds_config["text_column"]],
            "label": raw[ds_config["label_column"]],
        })
        splits[split] = map_labels(df, ds_config["label_map"])
        logger.info(f"Loaded {ds_config['hf_dataset_id']}/{filename}: {len(splits[split])} samples")
    return splits


def save_splits(splits: Dict[str, pd.DataFrame], out_dir: Union[str, Path]) -> None:
    """Save train/val/test DataFrames as CSV files and log their label balance."""
    out_dir = ensure_dir(out_dir)
    for split_name, split_df in splits.items():
        path = out_dir / f"{split_name}.csv"
        split_df[["text", "label"]].to_csv(path, index=False)
        pos_count = int((split_df["label"] == 1).sum())
        logger.info(f"Saved {split_name}: {len(split_df)} samples "
                    f"({pos_count} positive, {len(split_df) - pos_count} negative) to {path}")


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
    
    if args.use_demo:
        # Demo data has no predefined splits: standardize and split it here
        logger.info("Using demo datasets")
        demo_files = {"en": Path("data/raw/demo/demo_en.csv"),
                      "bn": Path("data/raw/demo/demo_bn.csv")}
        for lang, demo_path in demo_files.items():
            if args.language not in [lang, "all"]:
                continue
            df = standardize_labels(pd.read_csv(demo_path))
            split_and_save(
                df,
                Path(args.processed_dir) / lang,
                seed=seed,
                train_ratio=split_config["train_ratio"],
                val_ratio=split_config["val_ratio"],
                test_ratio=split_config["test_ratio"]
            )
    else:
        loaders = {"en": lambda: load_english_splits(config, seed),
                   "bn": lambda: load_bengali_splits(config)}
        for lang, load in loaders.items():
            if args.language not in [lang, "all"]:
                continue
            splits = load()
            stats = get_dataset_stats(pd.concat(splits.values()))
            logger.info(f"Dataset statistics for {lang}: {stats}")
            save_splits(splits, Path(args.processed_dir) / lang)
    
    logger.info("\nData loading complete!")


if __name__ == "__main__":
    main()
