"""
Tests for the results report
============================
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.report import collect_results, format_markdown


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _run(f1_en: float, f1_bn: float, seed: int) -> dict:
    return {
        "model": "FacebookAI/xlm-roberta-base",
        "seed": seed,
        "train_samples": 100,
        "test": {
            "en": {"n": 10, "accuracy": f1_en, "f1": f1_en},
            "bn": {"n": 20, "accuracy": f1_bn, "f1": f1_bn},
        },
    }


def test_seeds_are_averaged(tmp_path):
    _write(tmp_path / "zero_shot" / "seed1" / "metrics.json", _run(0.90, 0.60, 1))
    _write(tmp_path / "zero_shot" / "seed2" / "metrics.json", _run(0.80, 0.70, 2))
    
    results = collect_results(tmp_path)
    
    bn = results["transformer"]["zero_shot"]["test"]["bn"]["f1"]
    assert bn["runs"] == 2
    assert abs(bn["mean"] - 0.65) < 1e-9
    assert "65.0 ± 7.1" in format_markdown(results)


def test_baseline_without_transformer_runs(tmp_path):
    _write(tmp_path / "baseline" / "low_resource" / "metrics.json", {
        "train_samples": 50,
        "test": {
            "en": {"n": 10, "accuracy": 0.5, "f1": 0.33},
            "bn": {"n": 20, "accuracy": 0.75, "f1": 0.75},
        },
    })
    
    results = collect_results(tmp_path)
    
    assert results["transformer"] == {}
    assert "| low_resource | TF-IDF + LR | 33.0 | 75.0 |" in format_markdown(results)
