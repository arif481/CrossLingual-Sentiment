"""
Evaluation Module
=================

Evaluate trained models and generate comprehensive reports.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

from .utils import (
    load_config,
    set_seed,
    ensure_dir,
    save_json,
    load_json,
    get_device,
    setup_logging
)
from .preprocess import clean_text

logger = logging.getLogger(__name__)


def load_model_for_evaluation(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None
):
    """
    Load model for evaluation.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to use
        
    Returns:
        Tuple of (model, tokenizer, pipeline)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if device is None:
        device = get_device()
    
    logger.info(f"Loading model from {checkpoint_path}")
    
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Create pipeline
    classifier = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type == "cuda" else -1
    )
    
    return model, tokenizer, classifier


def evaluate_on_dataset(
    classifier,
    texts: List[str],
    labels: List[int],
    batch_size: int = 32,
    clean: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset.
    
    Args:
        classifier: Hugging Face pipeline
        texts: List of texts
        labels: List of true labels
        batch_size: Batch size for inference
        clean: Whether to clean texts
        
    Returns:
        Dictionary of metrics and predictions
    """
    if clean:
        texts = [clean_text(t) for t in texts]
    
    logger.info(f"Evaluating on {len(texts)} samples...")
    
    # Get predictions
    predictions = []
    scores = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = classifier(batch)
        
        for result in results:
            # Map label to int
            if result["label"].lower() in ["positive", "pos", "label_1"]:
                pred = 1
            else:
                pred = 0
            predictions.append(pred)
            scores.append(result["score"])
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    
    cm = confusion_matrix(labels, predictions)
    
    # Per-class metrics
    class_metrics = {
        "negative": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0])
        },
        "positive": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1])
        }
    }
    
    results = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": class_metrics,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(texts)
    }
    
    return results, predictions, scores


def evaluate_per_language(
    classifier,
    data_dir: Union[str, Path],
    languages: List[str],
    split: str = "test"
) -> Dict[str, Dict]:
    """
    Evaluate model separately on each language.
    
    Args:
        classifier: Hugging Face pipeline
        data_dir: Data directory
        languages: List of language codes
        split: Data split to evaluate
        
    Returns:
        Dictionary of results per language
    """
    data_dir = Path(data_dir)
    results = {}
    
    for lang in languages:
        test_path = data_dir / lang / f"{split}.csv"
        
        if not test_path.exists():
            logger.warning(f"File not found: {test_path}")
            continue
        
        logger.info(f"\nEvaluating on {lang.upper()} {split} set...")
        
        df = pd.read_csv(test_path)
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        
        lang_results, _, _ = evaluate_on_dataset(classifier, texts, labels)
        results[lang] = lang_results
        
        logger.info(f"  {lang.upper()} Results:")
        logger.info(f"    Accuracy: {lang_results['accuracy']:.4f}")
        logger.info(f"    Macro F1: {lang_results['macro_f1']:.4f}")
    
    return results


def generate_error_analysis(
    classifier,
    texts: List[str],
    labels: List[int],
    predictions: List[int],
    scores: List[float],
    n_samples: int = 50
) -> Dict[str, List]:
    """
    Generate error analysis report.
    
    Args:
        classifier: Hugging Face pipeline
        texts: Original texts
        labels: True labels
        predictions: Model predictions
        scores: Prediction confidence scores
        n_samples: Number of samples to analyze
        
    Returns:
        Dictionary with error analysis
    """
    errors = {
        "false_positives": [],
        "false_negatives": [],
        "low_confidence_correct": [],
        "high_confidence_wrong": []
    }
    
    for i, (text, label, pred, score) in enumerate(zip(texts, labels, predictions, scores)):
        if label != pred:
            error_info = {
                "text": text[:200],  # Truncate long texts
                "true_label": "positive" if label == 1 else "negative",
                "predicted": "positive" if pred == 1 else "negative",
                "confidence": score
            }
            
            if pred == 1 and label == 0:
                errors["false_positives"].append(error_info)
            elif pred == 0 and label == 1:
                errors["false_negatives"].append(error_info)
            
            if score > 0.9:
                errors["high_confidence_wrong"].append(error_info)
        else:
            if score < 0.6:
                errors["low_confidence_correct"].append({
                    "text": text[:200],
                    "label": "positive" if label == 1 else "negative",
                    "confidence": score
                })
    
    # Limit samples
    for key in errors:
        errors[key] = errors[key][:n_samples]
    
    return errors


def generate_html_report(
    results: Dict,
    output_path: Union[str, Path],
    mode: str = "unknown"
) -> None:
    """
    Generate HTML evaluation report.
    
    Args:
        results: Evaluation results dictionary
        output_path: Output HTML path
        mode: Experiment mode name
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report - {mode}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-size: 24px; color: #4CAF50; }}
        .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .card {{ background: #f9f9f9; padding: 20px; border-radius: 8px; min-width: 200px; }}
    </style>
</head>
<body>
    <h1>Cross-Lingual Sentiment Analysis - Evaluation Report</h1>
    <p><strong>Experiment Mode:</strong> {mode}</p>
    
    <h2>Overall Metrics</h2>
    <div class="container">
        <div class="card">
            <h3>Accuracy</h3>
            <p class="metric">{results.get('accuracy', 0):.2%}</p>
        </div>
        <div class="card">
            <h3>Macro F1</h3>
            <p class="metric">{results.get('macro_f1', 0):.2%}</p>
        </div>
        <div class="card">
            <h3>Precision</h3>
            <p class="metric">{results.get('macro_precision', 0):.2%}</p>
        </div>
        <div class="card">
            <h3>Recall</h3>
            <p class="metric">{results.get('macro_recall', 0):.2%}</p>
        </div>
    </div>
    
    <h2>Per-Class Metrics</h2>
    <table>
        <tr>
            <th>Class</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 Score</th>
            <th>Support</th>
        </tr>
"""
    
    if "per_class" in results:
        for class_name, metrics in results["per_class"].items():
            html += f"""
        <tr>
            <td>{class_name.capitalize()}</td>
            <td>{metrics['precision']:.4f}</td>
            <td>{metrics['recall']:.4f}</td>
            <td>{metrics['f1']:.4f}</td>
            <td>{metrics['support']}</td>
        </tr>
"""
    
    html += """
    </table>
    
    <h2>Confusion Matrix</h2>
    <table>
        <tr>
            <th></th>
            <th>Predicted Negative</th>
            <th>Predicted Positive</th>
        </tr>
"""
    
    if "confusion_matrix" in results:
        cm = results["confusion_matrix"]
        html += f"""
        <tr>
            <th>Actual Negative</th>
            <td>{cm[0][0]}</td>
            <td>{cm[0][1]}</td>
        </tr>
        <tr>
            <th>Actual Positive</th>
            <td>{cm[1][0]}</td>
            <td>{cm[1][1]}</td>
        </tr>
"""
    
    html += """
    </table>
    
    <p><em>Generated by Cross-Lingual Sentiment Analysis Framework</em></p>
</body>
</html>
"""
    
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    logger.info(f"HTML report saved to {output_path}")


def evaluate(
    checkpoint_path: Union[str, Path],
    data_dir: Union[str, Path],
    mode: str,
    config: Dict,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict:
    """
    Run complete evaluation pipeline.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to data directory
        mode: Experiment mode
        config: Configuration dictionary
        output_dir: Output directory for results
        
    Returns:
        Evaluation results
    """
    if output_dir is None:
        output_dir = Path(config["paths"]["output_dir"]) / mode
    output_dir = ensure_dir(output_dir)
    
    # Load model
    model, tokenizer, classifier = load_model_for_evaluation(checkpoint_path)
    
    # Get evaluation languages
    eval_langs = config["experiments"][mode]["eval_languages"]
    
    # Evaluate per language
    per_lang_results = evaluate_per_language(
        classifier, data_dir, eval_langs, split="test"
    )
    
    # Combined evaluation
    all_texts = []
    all_labels = []
    
    for lang in eval_langs:
        test_path = Path(data_dir) / lang / "test.csv"
        if test_path.exists():
            df = pd.read_csv(test_path)
            all_texts.extend(df["text"].tolist())
            all_labels.extend(df["label"].tolist())
    
    overall_results, predictions, scores = evaluate_on_dataset(
        classifier, all_texts, all_labels
    )
    
    # Error analysis
    error_analysis = generate_error_analysis(
        classifier, all_texts, all_labels, predictions, scores
    )
    
    # Compile results
    full_results = {
        "mode": mode,
        "checkpoint": str(checkpoint_path),
        "overall": overall_results,
        "per_language": per_lang_results,
        "error_analysis_summary": {
            "false_positives": len(error_analysis["false_positives"]),
            "false_negatives": len(error_analysis["false_negatives"]),
            "high_confidence_wrong": len(error_analysis["high_confidence_wrong"])
        }
    }
    
    # Save results
    save_json(full_results, output_dir / "eval_results.json")
    save_json(error_analysis, output_dir / "error_analysis.json")
    
    # Generate HTML report
    generate_html_report(overall_results, output_dir / "eval_report.html", mode)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Mode: {mode}")
    logger.info(f"Overall Accuracy: {overall_results['accuracy']:.4f}")
    logger.info(f"Overall Macro F1: {overall_results['macro_f1']:.4f}")
    
    if per_lang_results:
        logger.info("\nPer-Language Results:")
        for lang, results in per_lang_results.items():
            logger.info(f"  {lang.upper()}: Acc={results['accuracy']:.4f}, F1={results['macro_f1']:.4f}")
    
    return full_results


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate cross-lingual sentiment model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["zero_shot", "low_resource", "combined"],
        help="Experiment mode"
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
        help="Path to data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Evaluate on specific language only"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    
    # Override eval languages if specified
    if args.language:
        config["experiments"][args.mode]["eval_languages"] = [args.language]
    
    # Run evaluation
    results = evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        mode=args.mode,
        config=config,
        output_dir=args.output_dir
    )
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
