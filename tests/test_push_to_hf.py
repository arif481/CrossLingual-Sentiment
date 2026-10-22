"""
Tests for the model card
========================
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.push_to_hf import create_model_card

ROOT = Path(__file__).parent.parent


def test_model_card_uses_run_metrics():
    metrics = json.loads((ROOT / "experiments/combined/seed42/metrics.json").read_text())
    card = create_model_card("user/model", metrics)

    assert card.startswith("---\n")
    assert "base_model: FacebookAI/xlm-roberta-base" in card
    assert "fine-tuned on 73,662 examples" in card
    assert "| English: SST-2 validation split | 872 | 92.0 | 92.0 |" in card
    assert "| Bengali: SentNoB test split, neutral removed | 1,225 | 87.9 | 87.9 |" in card
    assert 'model="user/model"' in card
    assert "Trained for 4 epochs at learning rate 2e-5, batch size 32" in card
    assert "22 minutes on a Tesla T4" in card


def test_model_card_lists_only_training_languages():
    metrics = json.loads((ROOT / "experiments/zero_shot/seed42/metrics.json").read_text())
    card = create_model_card("user/model", metrics)

    assert "SST-2 movie-review sentences in English:" not in card
    assert "examples: SST-2 movie-review sentences in English." in card
    assert "SentNoB social media comments in Bengali." not in card
