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
) -> Dict[str, Dict[str, float]]:
    """
    Run baseline experiments for cross-lingual evaluation.
    
    Args:
        data_dir: Data directory
        output_dir: Output directory
        config: Configuration dict
        
    Returns:
        Dictionary of results per experiment
    """
    data_dir = Path(data_dir)
    results = {}
    
    # Experiment 1: Train on English, test on Bengali (Zero-shot)
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 1: Zero-shot (English → Bengali)")
    logger.info("="*60)
    
    en_train = data_dir / "en" / "train.csv"
    bn_test = data_dir / "bn" / "test.csv"
    
    if en_train.exists() and bn_test.exists():
        results["zero_shot"] = train_and_evaluate_baseline(
            en_train,
            bn_test,
            output_dir / "zero_shot",
            config["baseline"]
        )
    else:
        logger.warning("Missing files for zero-shot experiment")
    
    # Experiment 2: Train on Bengali, test on Bengali (Low-resource)
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 2: Low-resource (Bengali only)")
    logger.info("="*60)
    
    bn_train = data_dir / "bn" / "train.csv"
    
    if bn_train.exists() and bn_test.exists():
        results["low_resource"] = train_and_evaluate_baseline(
            bn_train,
            bn_test,
            output_dir / "low_resource",
            config["baseline"]
        )
    else:
        logger.warning("Missing files for low-resource experiment")
    
    # Experiment 3: Train on English, test on English (Monolingual)
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 3: Monolingual English")
    logger.info("="*60)
    
    en_test = data_dir / "en" / "test.csv"
    
    if en_train.exists() and en_test.exists():
        results["english_mono"] = train_and_evaluate_baseline(
            en_train,
            en_test,
            output_dir / "english_mono",
            config["baseline"]
        )
    else:
        logger.warning("Missing files for English monolingual experiment")
    
    # Save combined results
    save_json(results, output_dir / "all_baseline_results.json")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info("="*60)
    
    summary_rows = []
    for exp_name, metrics in results.items():
        summary_rows.append({
            "Experiment": exp_name,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "F1": f"{metrics['f1']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    logger.info(f"\n{summary_df.to_string(index=False)}")
    
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
