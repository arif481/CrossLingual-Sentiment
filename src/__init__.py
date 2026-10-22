"""
Cross-lingual sentiment analysis with XLM-RoBERTa: English (SST-2) to
Bengali (SentNoB) transfer, Bengali-only training and combined training.

Modules are imported on use (python -m src.train, from src import train)
so that data preparation does not load torch.
"""

__version__ = "0.2.0"
__author__ = "Md Arifuzzaman"

__all__ = [
    "baseline",
    "data_loader",
    "evaluate",
    "model",
    "preprocess",
    "report",
    "train",
    "utils",
]
