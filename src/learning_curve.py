"""
Learning Curve Module
=====================

How much does English training help when only a few Bengali labels exist?

For each seed, the Bengali training set is shuffled once and the first N
comments are used for every size, so the subsets are nested. Each subset
is used to fine-tune two starting points:

- bengali_only: XLM-RoBERTa straight from pre-training
- english_first: the model already fine-tuned on English SST-2 (the
  zero_shot run's best model)

Runs use a fixed schedule with no checkpoint selection, because a
1,213-comment validation set would be unrealistic next to 100 training
comments: 4 epochs or at least min_steps optimizer steps, whichever is
longer. Each finished run is written to
experiments/learning_curve/<condition>/n<N>/seed<seed>.json and skipped on
the next call, so an interrupted run can be resumed.
"""

import argparse
import gc
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)

from .model import build_model
from .preprocess import create_hf_dataset
from .train import evaluate_language
from .utils import load_config, load_json, save_json, set_seed, setup_logging

logger = logging.getLogger(__name__)

CONDITIONS = ["bengali_only", "english_first"]
CONDITION_NAMES = {
    "bengali_only": "XLM-R, Bengali only",
    "english_first": "XLM-R, English then Bengali",
}
DEFAULT_SIZES = [100, 250, 500, 1000, 2500, 9681]
DEFAULT_SEEDS = [42, 43, 44]


def subset_indices(n_total: int, size: int, seed: int) -> List[int]:
    """First `size` indices of a seeded permutation; larger sizes contain smaller ones."""
    order = np.random.RandomState(seed).permutation(n_total)
    return sorted(order[:min(size, n_total)].tolist())


def schedule(n_examples: int, batch_size: int, epochs: int, min_steps: int) -> int:
    """Number of epochs: `epochs`, or more if needed to reach `min_steps` steps."""
    steps_per_epoch = math.ceil(n_examples / batch_size)
    return max(epochs, math.ceil(min_steps / steps_per_epoch))


def run_one(
    condition: str,
    size: int,
    seed: int,
    config: Dict,
    tokenizer,
    bn_train,
    test_sets: Dict,
    english_model: Path,
    work_dir: Path,
    batch_size: int,
    epochs: int,
    min_steps: int
) -> Dict:
    """Fine-tune one starting point on one Bengali subset and test it."""
    set_seed(seed)
    train_ds = bn_train.select(subset_indices(len(bn_train), size, seed))
    n_epochs = schedule(len(train_ds), batch_size, epochs, min_steps)

    if condition == "bengali_only":
        model = build_model(
            model_name=config["model"]["default_model"],
            num_labels=config["model"]["num_labels"],
            dropout=config["model"]["dropout"],
            label2id=config["model"]["label2id"],
            id2label=config["model"]["id2label"]
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(english_model))

    train_config = config["training"]
    args = TrainingArguments(
        output_dir=str(work_dir),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=train_config["eval_batch_size"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        warmup_steps=train_config["warmup_ratio"],
        fp16=train_config["fp16"] and torch.cuda.is_available(),
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=seed,
        dataloader_num_workers=2
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer
    )

    start = time.time()
    trainer.train()
    result = {
        "condition": condition,
        "size": len(train_ds),
        "seed": seed,
        "positive_share": float(np.mean(list(train_ds["labels"]))),
        "epochs": n_epochs,
        "steps": int(trainer.state.global_step),
        "batch_size": batch_size,
        "learning_rate": train_config["learning_rate"],
        "start_model": config["model"]["default_model"] if condition == "bengali_only" else str(english_model),
        "test": {lang: evaluate_language(trainer, ds) for lang, ds in test_sets.items()},
        "runtime_seconds": time.time() - start,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    }

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def summarise(out_dir: Path, sizes: List[int]) -> Dict:
    """Average every finished run per condition and size."""
    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        for size in sizes:
            runs = [load_json(p) for p in sorted((out_dir / condition / f"n{size}").glob("seed*.json"))]
            if not runs:
                continue
            entry = {"runs": len(runs), "seeds": [r["seed"] for r in runs]}
            for lang in ["bn", "en"]:
                values = [r["test"][lang]["f1"] for r in runs]
                entry[f"{lang}_f1_mean"] = float(np.mean(values))
                entry[f"{lang}_f1_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                entry[f"{lang}_f1_runs"] = values
            summary[condition][str(size)] = entry
    return summary


def format_markdown(summary: Dict, sizes: List[int]) -> str:
    """Table of Bengali macro-F1 per size, plus English macro-F1 of the English-first model."""
    def cell(entry, lang):
        if entry is None:
            return "-"
        text = f"{100 * entry[f'{lang}_f1_mean']:.1f}"
        if entry["runs"] > 1:
            text += f" ± {100 * entry[f'{lang}_f1_std']:.1f}"
        return text

    lines = [
        "Macro-F1 (%) on the test sets, mean ± standard deviation over seeds",
        "",
        "| Bengali training comments | Bengali only: Bengali test | English then Bengali: Bengali test "
        "| Difference | English then Bengali: English test |",
        "|---|---|---|---|---|",
    ]
    for size in sizes:
        bn_only = summary["bengali_only"].get(str(size))
        en_first = summary["english_first"].get(str(size))
        diff = "-"
        if bn_only and en_first:
            diff = f"{100 * (en_first['bn_f1_mean'] - bn_only['bn_f1_mean']):+.1f}"
        lines.append(f"| {size:,} | {cell(bn_only, 'bn')} | {cell(en_first, 'bn')} | {diff} "
                     f"| {cell(en_first, 'en')} |")
    runs = sorted({e["runs"] for c in CONDITIONS for e in summary[c].values()})
    lines.append("")
    lines.append(f"Each cell averages {', '.join(map(str, runs))} run(s).")
    return "\n".join(lines) + "\n"


def plot(summary: Dict, sizes: List[int], path: Path, zero_shot_f1: Optional[float]) -> None:
    """Bengali macro-F1 against the number of Bengali training comments."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for condition, color in [("bengali_only", "#c2410c"), ("english_first", "#1d4ed8")]:
        points = [(s, summary[condition][str(s)]) for s in sizes if str(s) in summary[condition]]
        if not points:
            continue
        xs = [s for s, _ in points]
        means = np.array([100 * e["bn_f1_mean"] for _, e in points])
        stds = np.array([100 * e["bn_f1_std"] for _, e in points])
        ax.plot(xs, means, marker="o", color=color, label=CONDITION_NAMES[condition])
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.15, linewidth=0)
    if zero_shot_f1 is not None:
        ax.axhline(100 * zero_shot_f1, color="#6b7280", linestyle="--", linewidth=1)
        ax.annotate("English only, no Bengali labels", (sizes[0], 100 * zero_shot_f1),
                    xytext=(0, 4), textcoords="offset points", fontsize=8, color="#4b5563")
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.set_xlabel("Bengali training comments (log scale)")
    ax.set_ylabel("Bengali test macro-F1 (%)")
    ax.set_title("SentNoB, XLM-RoBERTa base (shading: ±1 std over seeds)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    """Run the learning curve and write its summary."""
    parser = argparse.ArgumentParser(description="Bengali learning curve with and without English training")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="experiments/learning_curve")
    parser.add_argument("--work-dir", type=str, default="models/learning_curve",
                        help="Scratch directory for the Trainer")
    parser.add_argument("--english-model", type=str, default="models/zero_shot-seed42/best",
                        help="English fine-tuned model to start english_first runs from")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Pretrained model for bengali_only runs (overrides config)")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--conditions", type=str, nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--min-steps", type=int, default=200)
    parser.add_argument("--max-eval-samples", type=int, default=None,
                        help="Cap on test samples, for quick checks")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only rebuild the summary from finished runs")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    if args.model_name is not None:
        config["model"]["default_model"] = args.model_name
    out_dir = Path(args.output_dir)
    english_model = Path(args.english_model)

    if not args.summary_only:
        if "english_first" in args.conditions and not english_model.exists():
            raise FileNotFoundError(
                f"{english_model} not found. Train it first with "
                "`python -m src.train --mode zero_shot --seed 42`."
            )
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["default_model"])
        max_length = config["model"]["max_length"]
        data_dir = Path(args.data_dir)
        bn_train = create_hf_dataset(data_dir / "bn" / "train.csv", tokenizer, max_length)
        test_sets = {}
        for lang in ["bn", "en"]:
            ds = create_hf_dataset(data_dir / lang / "test.csv", tokenizer, max_length)
            if args.max_eval_samples is not None:
                ds = ds.select(range(min(args.max_eval_samples, len(ds))))
            test_sets[lang] = ds

        for seed in args.seeds:
            for size in args.sizes:
                for condition in args.conditions:
                    path = out_dir / condition / f"n{size}" / f"seed{seed}.json"
                    if path.exists():
                        logger.info(f"{condition}, n={size}, seed {seed}: done already, skipping")
                        continue
                    logger.info(f"=== {condition}, n={size}, seed {seed} ===")
                    result = run_one(
                        condition, size, seed, config, tokenizer, bn_train, test_sets,
                        english_model, Path(args.work_dir), args.batch_size, args.epochs,
                        args.min_steps
                    )
                    save_json(result, path)
                    logger.info(f"  Bengali macro-F1={result['test']['bn']['f1']:.4f} "
                                f"English macro-F1={result['test']['en']['f1']:.4f} "
                                f"({result['steps']} steps, {result['runtime_seconds'] / 60:.1f} min)")

    summary = summarise(out_dir, args.sizes)
    zero_shot_path = Path(config["paths"]["output_dir"]) / "zero_shot" / "seed42" / "metrics.json"
    zero_shot_f1 = load_json(zero_shot_path)["test"]["bn"]["f1"] if zero_shot_path.exists() else None
    summary["zero_shot_bn_f1"] = zero_shot_f1
    save_json(summary, out_dir / "results.json")
    markdown = format_markdown(summary, args.sizes)
    (out_dir / "results.md").write_text(markdown, encoding="utf-8")
    plot(summary, args.sizes, out_dir / "learning_curve.png", zero_shot_f1)
    print(markdown)


if __name__ == "__main__":
    main()
