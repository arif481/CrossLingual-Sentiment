"""
Results Report Module
=====================

Collects the metrics.json files written by src.train and src.baseline and
builds experiments/results.json and experiments/results.md.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from .utils import load_json, save_json, setup_logging

logger = logging.getLogger(__name__)

MODES = ["zero_shot", "low_resource", "combined"]
LANGUAGES = ["en", "bn"]
LANGUAGE_NAMES = {"en": "English (SST-2)", "bn": "Bengali (SentNoB)"}


def _summarise(values: List[float]) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "runs": len(values)
    }


def collect_results(experiments_dir: Union[str, Path]) -> Dict:
    """Read every run and average test metrics over seeds."""
    experiments_dir = Path(experiments_dir)
    results = {"transformer": {}, "baseline": {}}

    for mode in MODES:
        baseline_path = experiments_dir / "baseline" / mode / "metrics.json"
        if baseline_path.exists():
            baseline = load_json(baseline_path)
            results["baseline"][mode] = {
                "train_samples": baseline["train_samples"],
                "test": {
                    lang: {key: baseline["test"][lang][key] for key in ["n", "accuracy", "f1"]}
                    for lang in LANGUAGES
                }
            }
        
        runs = [load_json(path) for path in sorted((experiments_dir / mode).glob("seed*/metrics.json"))]
        if runs:
            results["transformer"][mode] = {
                "model": runs[0]["model"],
                "seeds": [run["seed"] for run in runs],
                "train_samples": runs[0]["train_samples"],
                "test": {
                    lang: {
                        "n": runs[0]["test"][lang]["n"],
                        "accuracy": _summarise([run["test"][lang]["accuracy"] for run in runs]),
                        "f1": _summarise([run["test"][lang]["f1"] for run in runs])
                    }
                    for lang in LANGUAGES
                }
            }
    
    return results


def format_markdown(results: Dict) -> str:
    """Render the results as a Markdown table of macro-F1 and accuracy."""
    def cell(summary) -> str:
        if isinstance(summary, dict):
            text = f"{100 * summary['mean']:.1f}"
            if summary["runs"] > 1:
                text += f" ± {100 * summary['std']:.1f}"
            return text
        return f"{100 * summary:.1f}"

    lines = []
    for metric, title in [("f1", "Macro-F1"), ("accuracy", "Accuracy")]:
        lines.append(f"{title} (%) on each test set")
        lines.append("")
        lines.append("| Training data | Model | " + " | ".join(LANGUAGE_NAMES[l] for l in LANGUAGES) + " |")
        lines.append("|---|---|" + "---|" * len(LANGUAGES))
        for mode in MODES:
            if mode in results["baseline"]:
                row = results["baseline"][mode]["test"]
                lines.append(f"| {mode} | TF-IDF + LR | "
                             + " | ".join(cell(row[l][metric]) for l in LANGUAGES) + " |")
            if mode in results["transformer"]:
                entry = results["transformer"][mode]
                lines.append(f"| {mode} | {entry['model'].split('/')[-1]} | "
                             + " | ".join(cell(entry["test"][l][metric]) for l in LANGUAGES) + " |")
        lines.append("")

    seeds = sorted({len(entry["seeds"]) for entry in results["transformer"].values()})
    if seeds:
        lines.append(f"Transformer rows average {', '.join(map(str, seeds))} seed(s); "
                     "± is the standard deviation across seeds.")
    return "\n".join(lines) + "\n"


def main():
    """Build the results report."""
    parser = argparse.ArgumentParser(description="Summarise experiment results")
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Directory with experiment outputs"
    )
    args = parser.parse_args()

    setup_logging()

    results = collect_results(args.experiments_dir)
    markdown = format_markdown(results)
    save_json(results, Path(args.experiments_dir) / "results.json")
    (Path(args.experiments_dir) / "results.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
