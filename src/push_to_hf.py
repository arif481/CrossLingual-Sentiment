"""
Push to Hugging Face Hub Module
===============================

Upload trained models to Hugging Face Hub.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict

from huggingface_hub import (
    HfApi,
    create_repo,
    upload_folder,
    Repository
)

from .utils import load_config, load_json, setup_logging

logger = logging.getLogger(__name__)


def get_hf_token() -> str:
    """
    Get Hugging Face token from environment.
    
    Returns:
        HF token string
        
    Raises:
        ValueError if token not found
    """
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        # Try alternative env var names
        token = os.environ.get("HUGGINGFACE_TOKEN")
    
    if not token:
        # Try huggingface-cli token
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass
    
    if not token:
        raise ValueError(
            "Hugging Face token not found. Set HF_TOKEN environment variable "
            "or login with `huggingface-cli login`"
        )
    
    return token


def get_hf_username(token: str) -> str:
    """
    Get Hugging Face username from token.
    
    Args:
        token: HF token
        
    Returns:
        Username string
    """
    # First check environment
    username = os.environ.get("HF_USERNAME")
    
    if not username:
        # Get from API
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info["name"]
    
    return username


def create_model_card(
    model_name: str,
    languages: list,
    metrics: Optional[Dict] = None,
    mode: str = "combined"
) -> str:
    """
    Create a model card README.
    
    Args:
        model_name: Model name
        languages: List of languages
        metrics: Optional evaluation metrics
        mode: Training mode
        
    Returns:
        Model card string
    """
    metrics_str = ""
    if metrics:
        acc = metrics.get('accuracy', 'N/A')
        f1 = metrics.get('macro_f1', metrics.get('f1', 'N/A'))
        prec = metrics.get('macro_precision', metrics.get('precision', 'N/A'))
        rec = metrics.get('macro_recall', metrics.get('recall', 'N/A'))
        
        acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
        f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
        prec_str = f"{prec:.4f}" if isinstance(prec, (int, float)) else str(prec)
        rec_str = f"{rec:.4f}" if isinstance(rec, (int, float)) else str(rec)
        
        metrics_str = f"""
## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {acc_str} |
| Macro F1 | {f1_str} |
| Precision | {prec_str} |
| Recall | {rec_str} |
"""
    
    lang_str = ", ".join(languages)
    
    card = f"""---
language:
{chr(10).join(f'  - {lang}' for lang in languages)}
tags:
  - sentiment-analysis
  - cross-lingual
  - xlm-roberta
  - text-classification
datasets:
  - glue
  - sepidmnorozy/Bengali_sentiment
metrics:
  - accuracy
  - f1
library_name: transformers
pipeline_tag: text-classification
---

# {model_name}

A cross-lingual sentiment analysis model fine-tuned on XLM-RoBERTa for binary sentiment classification (positive/negative) across {lang_str}.

## Model Description

This model performs sentiment classification across multiple languages using transfer learning. It was trained using the **{mode}** strategy.

### Supported Languages
- English (en)
- Bengali (bn)

### Training Mode: {mode}
{_get_mode_description(mode)}

## Usage

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="{model_name}")

# English
result = classifier("This movie is absolutely fantastic!")
print(result)  # [{{'label': 'positive', 'score': 0.99}}]

# Bengali
result = classifier("এই সিনেমাটি অসাধারণ ছিল!")
print(result)  # [{{'label': 'positive', 'score': 0.95}}]
```

## Training

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
```

{metrics_str}

## Limitations

- Binary classification only (positive/negative)
- May not perform well on neutral sentiment
- Bengali performance may be lower than English due to limited training data

## Citation

If you use this model, please cite:

```bibtex
@misc{{crosslingual-sentiment,
  author = {{Cross-Lingual Sentiment Team}},
  title = {{Cross-Lingual Sentiment Analysis Model}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{model_name}}}
}}
```

## License

This model is released under the MIT License.
"""
    
    return card


def _get_mode_description(mode: str) -> str:
    """Get description for training mode."""
    descriptions = {
        "zero_shot": "Trained exclusively on English data and evaluated on Bengali without any Bengali training examples (zero-shot cross-lingual transfer).",
        "low_resource": "Trained only on Bengali data to establish a low-resource baseline.",
        "combined": "Trained on combined English and Bengali data for multilingual learning."
    }
    return descriptions.get(mode, "")


def push_to_hub(
    checkpoint_path: str,
    repo_name: str,
    mode: str = "combined",
    languages: list = None,
    metrics_path: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload model"
) -> str:
    """
    Push model checkpoint to Hugging Face Hub.
    
    Args:
        checkpoint_path: Path to model checkpoint
        repo_name: Repository name (without username)
        mode: Training mode
        languages: List of languages
        metrics_path: Optional path to metrics JSON
        private: Whether to make repo private
        commit_message: Git commit message
        
    Returns:
        Repository URL
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Get token and username
    token = get_hf_token()
    username = get_hf_username(token)
    
    full_repo_name = f"{username}/{repo_name}"
    logger.info(f"Pushing to: {full_repo_name}")
    
    # Default languages
    if languages is None:
        languages = ["en", "bn"]
    
    # Load metrics if available
    metrics = None
    if metrics_path and Path(metrics_path).exists():
        try:
            metrics_data = load_json(metrics_path)
            metrics = metrics_data.get("overall", metrics_data)
        except Exception as e:
            logger.warning(f"Could not load metrics: {e}")
    
    # Create model card
    model_card = create_model_card(
        model_name=full_repo_name,
        languages=languages,
        metrics=metrics,
        mode=mode
    )
    
    # Save model card to checkpoint
    readme_path = checkpoint_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    
    logger.info("Created model card")
    
    # Initialize API
    api = HfApi(token=token)
    
    # Create repository
    try:
        create_repo(
            repo_id=full_repo_name,
            token=token,
            private=private,
            exist_ok=True
        )
        logger.info(f"Repository created/verified: {full_repo_name}")
    except Exception as e:
        logger.warning(f"Repository creation note: {e}")
    
    # Upload files
    logger.info("Uploading model files...")
    
    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=full_repo_name,
        token=token,
        commit_message=commit_message
    )
    
    repo_url = f"https://huggingface.co/{full_repo_name}"
    logger.info(f"Successfully pushed to: {repo_url}")
    
    return repo_url


def main():
    """Main entry point for pushing to HF Hub."""
    parser = argparse.ArgumentParser(
        description="Push model to Hugging Face Hub"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="crosslingual-sentiment-model",
        help="Repository name (without username)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="combined",
        choices=["zero_shot", "low_resource", "combined"],
        help="Training mode for model card"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Path to metrics JSON file"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["en", "bn"],
        help="Languages supported by the model"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--message",
        type=str,
        default="Upload cross-lingual sentiment model",
        help="Commit message"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Check for token
    try:
        token = get_hf_token()
        logger.info("HF token found")
    except ValueError as e:
        logger.error(str(e))
        logger.error("\nTo set the token:")
        logger.error("  export HF_TOKEN=your_token_here")
        logger.error("  or: huggingface-cli login")
        return
    
    # Auto-detect metrics path
    metrics_path = args.metrics
    if not metrics_path:
        # Try to find metrics in checkpoint parent
        checkpoint_dir = Path(args.checkpoint).parent
        potential_paths = [
            checkpoint_dir / "eval_results.json",
            checkpoint_dir / "metrics.json",
            Path(args.checkpoint) / "metrics.json"
        ]
        for path in potential_paths:
            if path.exists():
                metrics_path = str(path)
                logger.info(f"Found metrics at: {metrics_path}")
                break
    
    # Push to hub
    try:
        repo_url = push_to_hub(
            checkpoint_path=args.checkpoint,
            repo_name=args.repo_name,
            mode=args.mode,
            languages=args.languages,
            metrics_path=metrics_path,
            private=args.private,
            commit_message=args.message
        )
        
        logger.info("\n" + "="*60)
        logger.info("SUCCESS!")
        logger.info("="*60)
        logger.info(f"Model available at: {repo_url}")
        logger.info("\nTo use the model:")
        logger.info(f'  classifier = pipeline("sentiment-analysis", model="{repo_url.split("/")[-2]}/{repo_url.split("/")[-1]}")')
        
    except Exception as e:
        logger.error(f"Failed to push model: {e}")
        raise


if __name__ == "__main__":
    main()
