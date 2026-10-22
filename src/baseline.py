"""
Baseline Models Module
======================

TF-IDF + Logistic Regression baseline for sentiment classification.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score

from .utils import load_config, ensure_dir, save_json, setup_logging
from .preprocess import clean_text

logger = logging.getLogger(__name__)


class TfidfBaseline:
    """
    TF-IDF + Logistic Regression baseline classifier.
    """
    
    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2),
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: str = "balanced"
    ):
        """
        Initialize baseline model.
        
        Args:
            max_features: Maximum vocabulary size
            ngram_range: N-gram range for TF-IDF
            solver: Logistic regression solver
            max_iter: Maximum iterations
            class_weight: Class weight strategy
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            strip_accents=None,  # Keep accents for multilingual
            lowercase=False,  # Don't lowercase (important for Bengali)
            sublinear_tf=True
        )
        
        self.classifier = LogisticRegression(
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42
        )
        
        self.is_fitted = False
    
    def fit(
        self,
        texts: List[str],
        labels: List[int],
        clean: bool = True
    ) -> "TfidfBaseline":
        """
        Fit the baseline model.
        
        Args:
            texts: List of text samples
            labels: List of labels
            clean: Whether to clean texts before training
            
        Returns:
            Self for chaining
        """
        if clean:
            texts = [clean_text(t) for t in texts]
        
        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} samples...")
        X = self.vectorizer.fit_transform(texts)
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logger.info(f"Feature matrix shape: {X.shape}")
        
        logger.info("Training logistic regression classifier...")
        self.classifier.fit(X, labels)
        
        self.is_fitted = True
        logger.info("Baseline model fitted successfully")
        
        return self
    
    def predict(
        self,
        texts: List[str],
        clean: bool = True
    ) -> np.ndarray:
        """
        Predict labels for texts.
        
        Args:
            texts: List of text samples
            clean: Whether to clean texts
            
        Returns:
            Array of predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if clean:
            texts = [clean_text(t) for t in texts]
        
        X = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X)
        
        return predictions
    
    def predict_proba(
        self,
        texts: List[str],
        clean: bool = True
    ) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            texts: List of text samples
            clean: Whether to clean texts
            
        Returns:
            Array of class probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if clean:
            texts = [clean_text(t) for t in texts]
        
        X = self.vectorizer.transform(texts)
        probas = self.classifier.predict_proba(X)
        
        return probas
    
    def evaluate(
        self,
        texts: List[str],
        labels: List[int],
        clean: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            texts: List of text samples
            labels: List of true labels
            clean: Whether to clean texts
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(texts, clean=clean)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro"
        )
        
        cm = confusion_matrix(labels, predictions)
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm.tolist()
        }
        
        return metrics
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.
        
        Args:
            path: Output path (directory)
        """
        path = ensure_dir(path)
        
        joblib.dump(self.vectorizer, path / "vectorizer.joblib")
        joblib.dump(self.classifier, path / "classifier.joblib")
        
        logger.info(f"Saved baseline model to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "TfidfBaseline":
        """
        Load model from disk.
        
        Args:
            path: Model directory
            
        Returns:
            Loaded TfidfBaseline instance
        """
        path = Path(path)
        
        model = cls()
        model.vectorizer = joblib.load(path / "vectorizer.joblib")
        model.classifier = joblib.load(path / "classifier.joblib")
        model.is_fitted = True
        
        logger.info(f"Loaded baseline model from {path}")
        
        return model


def train_and_evaluate_baseline(
    train_path: Union[str, Path],
    test_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Train and evaluate baseline model.
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        output_dir: Output directory
        config: Optional configuration dict
        
    Returns:
        Evaluation metrics
    """
    # Load config defaults
    if config is None:
        config = {
            "max_features": 50000,
            "ngram_range": [1, 2],
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": "balanced"
        }
    
    # Load data
    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    
    # Create and train model
    model = TfidfBaseline(
        max_features=config["max_features"],
        ngram_range=tuple(config["ngram_range"]),
        solver=config["solver"],
        max_iter=config["max_iter"],
        class_weight=config["class_weight"]
    )
    
    model.fit(
        train_df["text"].tolist(),
        train_df["label"].tolist()
    )
    
    # Evaluate
    logger.info("Evaluating on test set...")
    metrics = model.evaluate(
        test_df["text"].tolist(),
        test_df["label"].tolist()
    )
    
    # Print results
    logger.info("\nBaseline Results:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    
    # Full classification report
    predictions = model.predict(test_df["text"].tolist())
    report = classification_report(
        test_df["label"].tolist(),
        predictions,
        target_names=["Negative", "Positive"]
    )
    logger.info(f"\nClassification Report:\n{report}")
    
    # Save model and metrics
    output_dir = ensure_dir(output_dir)
    model.save(output_dir / "model")
    save_json(metrics, output_dir / "metrics.json")
    
    return metrics


def run_cross_lingual_baseline(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Dict
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run the baseline for every experiment mode, testing on both languages.
    
    Word features do not carry over between English and Bengali, so the
    zero-shot Bengali score shows what transfer looks like without a
    multilingual encoder.
    
    Args:
        data_dir: Data directory
        output_dir: Output directory
        config: Configuration dict
        
    Returns:
        Dictionary of test metrics per mode and language
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    baseline_config = config["baseline"]
    results = {}
    
    def read(lang: str, split: str) -> pd.DataFrame:
        return pd.read_csv(data_dir / lang / f"{split}.csv")
    
    for mode, exp_config in config["experiments"].items():
        train_langs = exp_config["train_languages"]
        logger.info("\n" + "=" * 60)
        logger.info(f"BASELINE: {mode} (train on {', '.join(train_langs)})")
        logger.info("=" * 60)
        
        train_df = pd.concat([read(lang, "train") for lang in train_langs])
        model = TfidfBaseline(
            max_features=baseline_config["max_features"],
            ngram_range=tuple(baseline_config["ngram_range"]),
            solver=baseline_config["solver"],
            max_iter=baseline_config["max_iter"],
            class_weight=baseline_config["class_weight"]
        )
        model.fit(train_df["text"].tolist(), train_df["label"].tolist())
        
        results[mode] = {
            "train_languages": train_langs,
            "train_samples": len(train_df),
            "sklearn_version": sklearn.__version__,
            "test": {}
        }
        for lang in ["en", "bn"]:
            test_df = read(lang, "test")
            metrics = model.evaluate(test_df["text"].tolist(), test_df["label"].tolist())
            metrics["n"] = len(test_df)
            results[mode]["test"][lang] = metrics
            logger.info(f"  {lang} test: accuracy={metrics['accuracy']:.4f} "
                        f"macro-F1={metrics['f1']:.4f}")
        
        save_json(results[mode], output_dir / mode / "metrics.json")
    
    save_json(results, output_dir / "all_baseline_results.json")
    return results


def main():
    """Main entry point for baseline experiments."""
    parser = argparse.ArgumentParser(
        description="Run TF-IDF + Logistic Regression baseline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed",
        help="Path to processed data directory"
    )
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Path to specific training CSV (overrides --data)"
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Path to specific test CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/baseline",
        help="Output directory"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all cross-lingual experiments"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load config
    config = load_config(args.config)
    
    if args.run_all:
        # Run all experiments
        run_cross_lingual_baseline(
            args.data,
            args.output,
            config
        )
    elif args.train and args.test:
        # Run single experiment
        train_and_evaluate_baseline(
            args.train,
            args.test,
            args.output,
            config.get("baseline", {})
        )
    else:
        parser.error("Either --run-all or both --train and --test are required")
    
    logger.info("\nBaseline experiments complete!")


if __name__ == "__main__":
    main()
