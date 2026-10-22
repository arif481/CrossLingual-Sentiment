"""
Push to Hugging Face Hub Module
===============================

Upload a trained model, with a model card built from its metrics.json,
and optionally the Gradio demo Space.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict

from huggingface_hub import CommitOperationAdd, HfApi, create_repo, get_token

from .utils import get_project_root, load_config, load_json, setup_logging

logger = logging.getLogger(__name__)

GITHUB_URL = "https://github.com/arif481/CrossLingual-Sentiment"

TRAINING_DATA = {
    "en": "SST-2 movie-review sentences in English",
    "bn": "SentNoB social media comments in Bengali"
}

TEST_SETS = {
    "en": "English: SST-2 validation split",
    "bn": "Bengali: SentNoB test split, neutral removed"
}

# Files the Trainer leaves next to the weights that a model repo doesn't need
IGNORE_PATTERNS = ["training_args.bin", "checkpoint-*", "*.pt", "*.pth"]


def get_hf_token() -> str:
    """
    Get Hugging Face token from the environment or the local login.

    Returns:
        HF token string

    Raises:
        ValueError if token not found
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or get_token()

    if not token:
        raise ValueError(
            "Hugging Face token not found. Set HF_TOKEN environment variable "
            "or login with `hf auth login`"
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


def create_model_card(model_id: str, metrics: Dict, max_length: int = 128) -> str:
    """
    Create a model card README from a metrics.json written by src.train.

    Args:
        model_id: Hub id of the model, e.g. user/name
        metrics: Contents of the run's metrics.json
        max_length: Maximum sequence length used in training

    Returns:
        Model card string
    """
    hp = metrics["hyperparameters"]
    versions = metrics["versions"]
    train_data = " and ".join(TRAINING_DATA[lang] for lang in metrics["train_languages"])
    learning_rate = f"{hp['learning_rate']:g}".replace("e-0", "e-")
    if metrics["epochs_trained"] < hp["epochs"]:
        epochs = f"{metrics['epochs_trained']:g} epochs (early stopping, out of up to {hp['epochs']})"
    else:
        epochs = f"{hp['epochs']} epochs"

    rows = "\n".join(
        f"| {TEST_SETS[lang]} | {result['n']:,} | {100 * result['accuracy']:.1f} | {100 * result['f1']:.1f} |"
        for lang, result in metrics["test"].items()
    )

    return f"""---
language:
  - en
  - bn
license: mit
base_model: {metrics["model"]}
tags:
  - sentiment-analysis
  - cross-lingual
  - xlm-roberta
  - text-classification
datasets:
  - nyu-mll/glue
  - khondoker/SentNoB
metrics:
  - accuracy
  - f1
library_name: transformers
pipeline_tag: text-classification
---

# XLM-RoBERTa base for English and Bengali sentiment

A negative/positive sentiment classifier. It is [{metrics["model"]}](https://huggingface.co/{metrics["model"]}) fine-tuned on {metrics["train_samples"]:,} examples: {train_data}. SentNoB's neutral comments were removed.

This is the `{metrics["mode"]}` model from [CrossLingual-Sentiment]({GITHUB_URL}). The project README compares it with zero-shot transfer from English, training on Bengali alone and a TF-IDF baseline.

## Results

One training run (seed {metrics["seed"]}):

| Test set | Examples | Accuracy (%) | Macro-F1 (%) |
|---|---|---|---|
{rows}

## Usage

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="{model_id}")
classifier(["The service was terrible and I'm very disappointed.", "এই পণ্যটি অসাধারণ! আমি এটা খুব পছন্দ করি।"])
```

The labels are `negative` and `positive`.

## Training

Trained for {epochs} at learning rate {learning_rate}, batch size {hp["batch_size"]}, weight decay {hp["weight_decay"]:g} and {100 * hp["warmup_ratio"]:.0f}% warmup, with inputs cut at {max_length} tokens. The checkpoint kept is the one with the best macro-F1 on the validation set ({metrics["val_samples"]:,} examples from the training languages). Training took {metrics["train_runtime_seconds"] / 60:.0f} minutes on a {metrics["device"]} with torch {versions["torch"]} and transformers {versions["transformers"]}.

## Limitations

- Two classes only. Neutral text still gets one of the two labels.
- The English data is movie reviews and the Bengali data is social media comments. Other kinds of text were not tested.
- The scores come from one training run. With {metrics["test"]["bn"]["n"]:,} Bengali test comments, differences of a point or two between models are noise.

## License

The model and code are MIT licensed. SentNoB (Islam et al., Findings of EMNLP 2021) is licensed CC BY-ND 4.0; check its terms before reusing the data.
"""


def push_to_hub(
    checkpoint_path: str,
    repo_name: str,
    metrics_path: str,
    max_length: int = 128,
    private: bool = False,
    commit_message: str = "Upload model"
) -> str:
    """
    Push a saved model to the Hugging Face Hub, replacing what the repo held.

    Args:
        checkpoint_path: Directory with the saved model and tokenizer
        repo_name: Repository name (without username)
        metrics_path: The run's metrics.json, used for the model card
        max_length: Maximum sequence length used in training
        private: Whether to make repo private
        commit_message: Commit message on the Hub

    Returns:
        Repository URL
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    token = get_hf_token()
    full_repo_name = f"{get_hf_username(token)}/{repo_name}"
    logger.info(f"Pushing to: {full_repo_name}")

    model_card = create_model_card(full_repo_name, load_json(metrics_path), max_length)
    (checkpoint_path / "README.md").write_text(model_card, encoding="utf-8")
    logger.info("Created model card")

    create_repo(repo_id=full_repo_name, token=token, private=private, exist_ok=True)

    # delete_patterns="*" removes files from earlier uploads that this one
    # doesn't replace; the Hub always keeps .gitattributes
    HfApi(token=token).upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=full_repo_name,
        ignore_patterns=IGNORE_PATTERNS,
        delete_patterns="*",
        commit_message=commit_message
    )

    repo_url = f"https://huggingface.co/{full_repo_name}"
    logger.info(f"Successfully pushed to: {repo_url}")

    return repo_url


def push_space(space_name: str, commit_message: str = "Update demo") -> str:
    """
    Upload the Gradio demo in app/ to a Hugging Face Space.

    Args:
        space_name: Space name (without username)
        commit_message: Commit message on the Hub

    Returns:
        Space URL
    """
    token = get_hf_token()
    space_id = f"{get_hf_username(token)}/{space_name}"
    app_dir = get_project_root() / "app"

    files = {
        "app.py": app_dir / "gradio_app.py",
        "requirements.txt": app_dir / "requirements_hf.txt",
        "README.md": app_dir / "space_README.md"
    }
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
        for name, path in files.items()
    ]
    HfApi(token=token).create_commit(
        repo_id=space_id,
        repo_type="space",
        operations=operations,
        commit_message=commit_message
    )

    space_url = f"https://huggingface.co/spaces/{space_id}"
    logger.info(f"Successfully pushed to: {space_url}")

    return space_url


def main():
    """Main entry point for pushing to HF Hub."""
    parser = argparse.ArgumentParser(
        description="Push model to Hugging Face Hub"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/combined-seed42/best",
        help="Directory with the saved model"
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
        help="Training mode, used to find the metrics file"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Path to the run's metrics.json (default: experiments/<mode>/seed42/metrics.json)"
    )
    parser.add_argument(
        "--space",
        type=str,
        default=None,
        help="Also upload the Gradio demo to this Space (name without username)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
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

    setup_logging()

    try:
        get_hf_token()
    except ValueError as e:
        logger.error(str(e))
        return

    metrics_path = args.metrics or f"experiments/{args.mode}/seed42/metrics.json"
    if not Path(metrics_path).exists():
        logger.error(f"Metrics not found: {metrics_path}")
        return

    config = load_config(args.config)
    push_to_hub(
        checkpoint_path=args.checkpoint,
        repo_name=args.repo_name,
        metrics_path=metrics_path,
        max_length=config["model"]["max_length"],
        private=args.private,
        commit_message=args.message
    )

    if args.space:
        push_space(args.space)


if __name__ == "__main__":
    main()
