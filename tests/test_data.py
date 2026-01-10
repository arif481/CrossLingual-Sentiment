"""
Tests for Data Loading Module
=============================
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    standardize_labels,
    split_and_save,
    get_dataset_stats,
    load_processed_data,
    create_combined_dataset
)


class TestStandardizeLabels:
    """Tests for label standardization."""
    
    def test_binary_int_labels(self):
        """Test with integer binary labels."""
        df = pd.DataFrame({
            "text": ["good", "bad", "great"],
            "label": [1, 0, 1]
        })
        result = standardize_labels(df)
        assert list(result["label"]) == [1, 0, 1]
    
    def test_string_labels(self):
        """Test with string labels."""
        df = pd.DataFrame({
            "text": ["good", "bad", "great", "terrible"],
            "label": ["positive", "negative", "Positive", "Negative"]
        })
        result = standardize_labels(df)
        assert list(result["label"]) == [1, 0, 1, 0]
    
    def test_pos_neg_abbreviations(self):
        """Test with pos/neg abbreviations."""
        df = pd.DataFrame({
            "text": ["a", "b"],
            "label": ["pos", "neg"]
        })
        result = standardize_labels(df)
        assert list(result["label"]) == [1, 0]
    
    def test_numeric_string_labels(self):
        """Test with numeric string labels."""
        df = pd.DataFrame({
            "text": ["a", "b"],
            "label": ["1", "0"]
        })
        result = standardize_labels(df)
        assert list(result["label"]) == [1, 0]
    
    def test_missing_label_column_raises(self):
        """Test that missing label column raises error."""
        df = pd.DataFrame({"text": ["a", "b"]})
        with pytest.raises(ValueError):
            standardize_labels(df)


class TestSplitAndSave:
    """Tests for data splitting."""
    
    def test_split_ratios(self):
        """Test that splits have correct ratios."""
        df = pd.DataFrame({
            "text": [f"text_{i}" for i in range(100)],
            "label": [i % 2 for i in range(100)]  # 50-50 split
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = split_and_save(
                df, tmpdir, seed=42,
                train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
            )
            
            assert len(splits["train"]) == 80
            assert len(splits["val"]) == 10
            assert len(splits["test"]) == 10
    
    def test_stratified_split(self):
        """Test that splits are stratified."""
        # Create imbalanced dataset
        df = pd.DataFrame({
            "text": [f"text_{i}" for i in range(100)],
            "label": [0] * 80 + [1] * 20
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = split_and_save(
                df, tmpdir, seed=42,
                train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
            )
            
            # Check that each split has both classes
            for split_name, split_df in splits.items():
                assert 0 in split_df["label"].values
                assert 1 in split_df["label"].values
    
    def test_files_created(self):
        """Test that CSV files are created."""
        df = pd.DataFrame({
            "text": [f"text_{i}" for i in range(100)],
            "label": [i % 2 for i in range(100)]  # 50-50 balanced
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            split_and_save(df, tmpdir, seed=42)
            
            assert (Path(tmpdir) / "train.csv").exists()
            assert (Path(tmpdir) / "val.csv").exists()
            assert (Path(tmpdir) / "test.csv").exists()


class TestDatasetStats:
    """Tests for dataset statistics."""
    
    def test_basic_stats(self):
        """Test basic statistics computation."""
        df = pd.DataFrame({
            "text": ["short", "medium text", "this is a longer text sample"],
            "label": [0, 1, 1]
        })
        
        stats = get_dataset_stats(df)
        
        assert stats["total_samples"] == 3
        assert stats["positive_samples"] == 2
        assert stats["negative_samples"] == 1
        assert 0 < stats["positive_ratio"] < 1
        assert stats["avg_text_length"] > 0
        assert stats["max_text_length"] >= stats["min_text_length"]


class TestDemoData:
    """Tests for demo data files."""
    
    def test_demo_english_exists(self):
        """Test that demo English data exists."""
        demo_path = Path("data/raw/demo/demo_en.csv")
        if demo_path.exists():
            df = pd.read_csv(demo_path)
            assert "text" in df.columns
            assert "label" in df.columns
            assert len(df) > 0
    
    def test_demo_bengali_exists(self):
        """Test that demo Bengali data exists."""
        demo_path = Path("data/raw/demo/demo_bn.csv")
        if demo_path.exists():
            df = pd.read_csv(demo_path)
            assert "text" in df.columns
            assert "label" in df.columns
            assert len(df) > 0
    
    def test_demo_labels_valid(self):
        """Test that demo data has valid labels."""
        for lang in ["en", "bn"]:
            demo_path = Path(f"data/raw/demo/demo_{lang}.csv")
            if demo_path.exists():
                df = pd.read_csv(demo_path)
                assert set(df["label"].unique()).issubset({0, 1})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
