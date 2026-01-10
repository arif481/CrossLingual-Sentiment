"""
Training Module
===============

Training script for cross-lingual sentiment analysis using
Hugging Face Trainer API.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, List, Union, Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .utils import (
    load_config,
    set_seed,
    ensure_dir,
    save_json,
    get_device,
    setup_logging
)
from .preprocess import create_hf_dataset, clean_dataframe
from .model import build_model

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for Trainer.
    
    Args:
        eval_pred: EvalPrediction with predictions and labels
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def load_dataset_for_mode(
    mode: str,
    data_dir: Union[str, Path],
    tokenizer,
    max_length: int,
    config: Dict
) -> Dict[str, Dataset]:
    """
    Load datasets based on experiment mode.
    
    Args:
        mode: Experiment mode (zero_shot, low_resource, combined)
        data_dir: Data directory path
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        config: Configuration dictionary
        
    Returns:
        Dictionary with train, val, test datasets
    """
    data_dir = Path(data_dir)
    datasets = {}
    
    exp_config = config["experiments"][mode]
    train_langs = exp_config["train_languages"]
    eval_langs = exp_config["eval_languages"]
    
    logger.info(f"Mode: {mode}")
    logger.info(f"  Training languages: {train_langs}")
    logger.info(f"  Evaluation languages: {eval_langs}")
    
    # Load training data
    train_datasets = []
    for lang in train_langs:
        train_path = data_dir / lang / "train.csv"
        if train_path.exists():
            ds = create_hf_dataset(train_path, tokenizer, max_length)
            train_datasets.append(ds)
            logger.info(f"  Loaded {lang} train: {len(ds)} samples")
        else:
            logger.warning(f"  Train file not found: {train_path}")
    
    if train_datasets:
        datasets["train"] = concatenate_datasets(train_datasets)
        datasets["train"] = datasets["train"].shuffle(seed=config["training"]["seed"])
        logger.info(f"  Total training samples: {len(datasets['train'])}")
    
    # Load validation data (use eval languages)
    val_datasets = []
    for lang in eval_langs:
        val_path = data_dir / lang / "val.csv"
        if val_path.exists():
            ds = create_hf_dataset(val_path, tokenizer, max_length)
            val_datasets.append(ds)
            logger.info(f"  Loaded {lang} val: {len(ds)} samples")
    
    if val_datasets:
        datasets["val"] = concatenate_datasets(val_datasets)
        logger.info(f"  Total validation samples: {len(datasets['val'])}")
    
    # Load test data (use eval languages)
    test_datasets = []
    for lang in eval_langs:
        test_path = data_dir / lang / "test.csv"
        if test_path.exists():
            ds = create_hf_dataset(test_path, tokenizer, max_length)
            test_datasets.append(ds)
            logger.info(f"  Loaded {lang} test: {len(ds)} samples")
    
    if test_datasets:
        datasets["test"] = concatenate_datasets(test_datasets)
        logger.info(f"  Total test samples: {len(datasets['test'])}")
    
    return datasets


def train(
    mode: str,
    config: Dict,
    data_dir: Union[str, Path] = "data/processed",
    output_dir: Optional[Union[str, Path]] = None,
    resume_from: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train model for specified experiment mode.
    
    Args:
        mode: Experiment mode (zero_shot, low_resource, combined)
        config: Configuration dictionary
        data_dir: Path to processed data
        output_dir: Output directory (defaults to experiments/<mode>)
        resume_from: Optional checkpoint to resume from
        
    Returns:
        Training results and metrics
    """
    # Set seed for reproducibility
    seed = config["training"]["seed"]
    set_seed(seed)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(config["paths"]["output_dir"]) / mode
    output_dir = ensure_dir(output_dir)
    
    logger.info(f"Starting training: mode={mode}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load tokenizer
    model_name = config["model"]["default_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load datasets
    max_length = config["model"]["max_length"]
    datasets = load_dataset_for_mode(mode, data_dir, tokenizer, max_length, config)
    
    if "train" not in datasets:
        raise ValueError("No training data available")
    
    # Build model
    model = build_model(
        model_name=model_name,
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
        label2id=config["model"]["label2id"],
        id2label=config["model"]["id2label"]
    )
    
    # Setup training arguments
    train_config = config["training"]
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config["epochs"],
        per_device_train_batch_size=train_config["batch_size"],
        per_device_eval_batch_size=train_config["eval_batch_size"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        warmup_ratio=train_config["warmup_ratio"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        fp16=train_config["fp16"] and torch.cuda.is_available(),
        evaluation_strategy="steps",
        eval_steps=train_config["eval_steps"],
        save_strategy="steps",
        save_steps=train_config["save_steps"],
        save_total_limit=train_config["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model=train_config["metric_for_best_model"],
        greater_is_better=train_config["greater_is_better"],
        logging_dir=str(output_dir / "logs"),
        logging_steps=train_config["logging_steps"],
        report_to=["tensorboard"],
        seed=seed,
        dataloader_num_workers=2,
        remove_unused_columns=True
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True
    )
    
    # Callbacks
    callbacks = []
    if train_config["early_stopping_patience"] > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=train_config["early_stopping_patience"]
            )
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("val"),
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    
    # Train
    logger.info("Starting training...")
    
    if resume_from:
        train_result = trainer.train(resume_from_checkpoint=resume_from)
    else:
        train_result = trainer.train()
    
    # Save best model
    best_model_dir = output_dir / "checkpoint-best"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    
    logger.info(f"Best model saved to {best_model_dir}")
    
    # Evaluate on test set
    results = {"train_metrics": train_result.metrics}
    
    if "test" in datasets:
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(datasets["test"], metric_key_prefix="test")
        results["test_metrics"] = test_results
        
        logger.info("Test Results:")
        for key, value in test_results.items():
            logger.info(f"  {key}: {value:.4f}")
    
    # Save all metrics
    save_json(results, output_dir / "metrics.json")
    
    # Save training history
    if trainer.state.log_history:
        save_json(trainer.state.log_history, output_dir / "training_history.json")
    
    return results


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train cross-lingual sentiment model"
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
        help="Path to processed data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint to resume from"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with minimal training"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load config
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    
    # Demo mode: quick training for testing
    if args.demo:
        logger.info("Running in DEMO mode with minimal settings")
        config["training"]["epochs"] = 1
        config["training"]["eval_steps"] = 10
        config["training"]["save_steps"] = 10
        config["training"]["logging_steps"] = 5
    
    # Run training
    results = train(
        mode=args.mode,
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resume_from=args.resume_from
    )
    
    logger.info("\nTraining complete!")
    logger.info(f"Results saved to: {args.output_dir or config['paths']['output_dir']}/{args.mode}")


if __name__ == "__main__":
    main()
