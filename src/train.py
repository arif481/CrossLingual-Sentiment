"""
Training Module
===============

Training script for cross-lingual sentiment analysis using
Hugging Face Trainer API.

Each run trains on the mode's training languages, picks the best epoch on
validation data from those same languages, and reports test metrics for
English and Bengali separately. The Bengali validation set is never seen in
zero-shot mode.
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Union, Any

import datasets as hf_datasets
import numpy as np
import torch
import transformers
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)

from .utils import (
    load_config,
    set_seed,
    ensure_dir,
    save_json,
    setup_logging
)
from .preprocess import create_hf_dataset
from .model import build_model

logger = logging.getLogger(__name__)

LANGUAGES = ["en", "bn"]


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for Trainer.

    Args:
        eval_pred: EvalPrediction with predictions and labels

    Returns:
        Dictionary of metrics (precision, recall and F1 are macro-averaged)
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def _limit(dataset: Dataset, max_samples: Optional[int]) -> Dataset:
    """Keep the first max_samples rows (for quick smoke runs)."""
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return dataset.select(range(max_samples))


def load_dataset_for_mode(
    mode: str,
    data_dir: Union[str, Path],
    tokenizer,
    max_length: int,
    config: Dict,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Load datasets based on experiment mode.

    Args:
        mode: Experiment mode (zero_shot, low_resource, combined)
        data_dir: Data directory path
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        config: Configuration dictionary
        max_train_samples: Optional cap on training samples
        max_eval_samples: Optional cap on validation and test samples

    Returns:
        Dictionary with "train" and "val" datasets (training languages) and
        "test", a dict of test datasets keyed by language
    """
    data_dir = Path(data_dir)
    seed = config["training"]["seed"]
    train_langs = config["experiments"][mode]["train_languages"]

    logger.info(f"Mode: {mode}")
    logger.info(f"  Training and validation languages: {train_langs}")

    def load_split(lang: str, split: str) -> Dataset:
        path = data_dir / lang / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.data_loader` first."
            )
        ds = create_hf_dataset(path, tokenizer, max_length)
        logger.info(f"  Loaded {lang} {split}: {len(ds)} samples")
        return ds

    train_ds = concatenate_datasets([load_split(lang, "train") for lang in train_langs])
    val_ds = concatenate_datasets([load_split(lang, "val") for lang in train_langs])

    datasets = {
        "train": _limit(train_ds.shuffle(seed=seed), max_train_samples),
        "val": _limit(val_ds.shuffle(seed=seed), max_eval_samples),
        "test": {
            lang: _limit(load_split(lang, "test"), max_eval_samples)
            for lang in LANGUAGES
        }
    }
    logger.info(f"  Training samples: {len(datasets['train'])}")
    logger.info(f"  Validation samples: {len(datasets['val'])}")

    return datasets


def evaluate_language(trainer: Trainer, dataset: Dataset) -> Dict[str, Any]:
    """Predict on one test set and return metrics plus the confusion matrix."""
    output = trainer.predict(dataset)
    predictions = np.argmax(output.predictions, axis=1)
    metrics = compute_metrics((output.predictions, output.label_ids))
    metrics = {key: float(value) for key, value in metrics.items()}
    metrics["n"] = int(len(dataset))
    # Rows are true labels, columns are predictions: [[TN, FP], [FN, TP]]
    metrics["confusion_matrix"] = confusion_matrix(
        output.label_ids, predictions, labels=[0, 1]
    ).tolist()
    return metrics


def train(
    mode: str,
    config: Dict,
    data_dir: Union[str, Path] = "data/processed",
    output_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Train model for specified experiment mode.

    Args:
        mode: Experiment mode (zero_shot, low_resource, combined)
        config: Configuration dictionary
        data_dir: Path to processed data
        output_dir: Where metrics go (defaults to experiments/<mode>/seed<seed>)
        models_dir: Where checkpoints and the final model go
            (defaults to models/<mode>-seed<seed>)
        max_train_samples: Optional cap on training samples
        max_eval_samples: Optional cap on validation and test samples
        save_model: Whether to keep the best model after training

    Returns:
        Training results and metrics
    """
    # Set seed for reproducibility
    seed = config["training"]["seed"]
    set_seed(seed)

    run_name = f"{mode}-seed{seed}"
    if output_dir is None:
        output_dir = Path(config["paths"]["output_dir"]) / mode / f"seed{seed}"
    if models_dir is None:
        models_dir = Path(config["paths"]["models_dir"]) / run_name
    output_dir = ensure_dir(output_dir)
    models_dir = ensure_dir(models_dir)

    logger.info(f"Starting training: {run_name}")
    logger.info(f"Metrics directory: {output_dir}")
    logger.info(f"Model directory: {models_dir}")

    # Load tokenizer
    model_name = config["model"]["default_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load datasets
    max_length = config["model"]["max_length"]
    datasets = load_dataset_for_mode(
        mode, data_dir, tokenizer, max_length, config,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples
    )

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
        output_dir=str(models_dir),
        num_train_epochs=train_config["epochs"],
        per_device_train_batch_size=train_config["batch_size"],
        per_device_eval_batch_size=train_config["eval_batch_size"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        # A float below 1 is read as a fraction of the total training steps
        warmup_steps=train_config["warmup_ratio"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        fp16=train_config["fp16"] and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="best",
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model=train_config["metric_for_best_model"],
        greater_is_better=train_config["greater_is_better"],
        logging_steps=train_config["logging_steps"],
        report_to="none",
        seed=seed,
        dataloader_num_workers=2
    )

    # Pad each batch to its longest sequence
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
        eval_dataset=datasets["val"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Evaluate the best epoch on each language's test set
    test_metrics = {}
    for lang, test_ds in datasets["test"].items():
        logger.info(f"Evaluating on {lang} test set...")
        test_metrics[lang] = evaluate_language(trainer, test_ds)
        logger.info(f"  {lang}: accuracy={test_metrics[lang]['accuracy']:.4f} "
                    f"macro-F1={test_metrics[lang]['f1']:.4f}")

    results = {
        "mode": mode,
        "seed": seed,
        "model": model_name,
        "train_languages": config["experiments"][mode]["train_languages"],
        "train_samples": len(datasets["train"]),
        "val_samples": len(datasets["val"]),
        "epochs_trained": float(trainer.state.epoch),
        "best_val_f1": trainer.state.best_metric,
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": hf_datasets.__version__
        },
        "hyperparameters": {
            key: train_config[key]
            for key in ["epochs", "batch_size", "learning_rate", "weight_decay",
                        "warmup_ratio", "early_stopping_patience"]
        },
        "test": test_metrics
    }

    # Keep only the best model; intermediate checkpoints take ~1 GB each
    for checkpoint in models_dir.glob("checkpoint-*"):
        shutil.rmtree(checkpoint)
    if save_model:
        best_model_dir = models_dir / "best"
        trainer.save_model(str(best_model_dir))
        tokenizer.save_pretrained(str(best_model_dir))
        logger.info(f"Best model saved to {best_model_dir}")

    # Save all metrics
    save_json(results, output_dir / "metrics.json")
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
        help="Metrics directory (default: experiments/<mode>/seed<seed>)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Checkpoint and model directory (default: models/<mode>-seed<seed>)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Pretrained model (overrides config)"
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
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap on training samples, for quick checks"
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Cap on validation and test samples, for quick checks"
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Do not keep the trained model"
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
    if args.model_name is not None:
        config["model"]["default_model"] = args.model_name
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
        config["training"]["logging_steps"] = 5

    # Run training
    train(
        mode=args.mode,
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        save_model=not args.no_save_model
    )

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
