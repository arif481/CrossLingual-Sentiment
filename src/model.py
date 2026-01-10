"""
Model Module
============

Hugging Face model wrapper for cross-lingual sentiment classification.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer
)

from .utils import set_seed, get_device

logger = logging.getLogger(__name__)


def build_model(
    model_name: str = "xlm-roberta-base",
    num_labels: int = 2,
    dropout: float = 0.1,
    label2id: Optional[Dict[str, int]] = None,
    id2label: Optional[Dict[int, str]] = None
) -> AutoModelForSequenceClassification:
    """
    Build a Hugging Face model for sequence classification.
    
    Args:
        model_name: Pretrained model name or path
        num_labels: Number of classification labels
        dropout: Dropout probability
        label2id: Mapping from label names to IDs
        id2label: Mapping from IDs to label names
        
    Returns:
        AutoModelForSequenceClassification instance
    """
    # Default label mappings
    if label2id is None:
        label2id = {"negative": 0, "positive": 1}
    if id2label is None:
        id2label = {0: "negative", 1: "positive"}
    
    logger.info(f"Building model: {model_name}")
    logger.info(f"  num_labels: {num_labels}")
    logger.info(f"  dropout: {dropout}")
    
    # Load config and modify
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        label2id=label2id,
        id2label=id2label
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model loaded successfully")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load a model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    if device is None:
        device = get_device()
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Checkpoint loaded on {device}")
    
    return model, tokenizer


def save_checkpoint(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    output_dir: Union[str, Path],
    metrics: Optional[Dict] = None
) -> None:
    """
    Save model checkpoint with optional metrics.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
        metrics: Optional metrics dictionary
    """
    from .utils import ensure_dir, save_json
    
    output_dir = ensure_dir(output_dir)
    
    logger.info(f"Saving checkpoint to {output_dir}")
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metrics if provided
    if metrics:
        save_json(metrics, output_dir / "metrics.json")
    
    logger.info("Checkpoint saved successfully")


class SentimentClassifier(nn.Module):
    """
    Custom wrapper for sentiment classification with additional features.
    """
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_base: bool = False
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: Pretrained model name
            num_labels: Number of labels
            dropout: Dropout probability
            freeze_base: Whether to freeze base model
        """
        super().__init__()
        
        self.model = build_model(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout
        )
        
        if freeze_base:
            self._freeze_base_model()
    
    def _freeze_base_model(self) -> None:
        """Freeze base model parameters."""
        # XLM-RoBERTa has 'roberta' as base
        if hasattr(self.model, 'roberta'):
            for param in self.model.roberta.parameters():
                param.requires_grad = False
            logger.info("Base model frozen")
        elif hasattr(self.model, 'bert'):
            for param in self.model.bert.parameters():
                param.requires_grad = False
            logger.info("Base model frozen")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Optional labels for training
            
        Returns:
            Model outputs
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get predictions without labels.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Predicted class indices
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        return predictions
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Class probabilities
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probas = torch.softmax(outputs.logits, dim=-1)
        
        return probas


def create_dummy_model(
    output_dir: Union[str, Path],
    model_name: str = "xlm-roberta-base"
) -> None:
    """
    Create a dummy model for demo purposes.
    
    This creates a model with random weights that can be used
    to test the inference pipeline without training.
    
    Args:
        output_dir: Output directory
        model_name: Model architecture to use
    """
    from .utils import ensure_dir
    
    output_dir = ensure_dir(output_dir)
    
    logger.info(f"Creating dummy model at {output_dir}")
    
    # Create model and tokenizer
    model = build_model(model_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Create a README
    readme = """# Demo Checkpoint

This is a demonstration checkpoint with random/untrained weights.
It is meant for testing the inference pipeline.

To use a real model, either:
1. Train your own model using `python -m src.train`
2. Download a pretrained model from Hugging Face Hub

## Usage

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="path/to/checkpoint"
)

result = classifier("This movie is great!")
```
"""
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)
    
    logger.info("Dummy model created successfully")


if __name__ == "__main__":
    import argparse
    from .utils import setup_logging
    
    parser = argparse.ArgumentParser(description="Model utilities")
    parser.add_argument(
        "--create-dummy",
        action="store_true",
        help="Create a dummy model for testing"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/demo-checkpoint",
        help="Output directory for dummy model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="xlm-roberta-base",
        help="Model architecture"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    set_seed(42)
    
    if args.create_dummy:
        create_dummy_model(args.output, args.model_name)
