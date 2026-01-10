"""
Cross-Lingual Sentiment Analysis Package
========================================

A research-ready framework for cross-lingual sentiment analysis
using multilingual transformers (XLM-RoBERTa).

Supports:
- Zero-shot cross-lingual transfer (English → Bengali)
- Low-resource fine-tuning
- Combined multilingual training
"""

__version__ = "0.1.0"
__author__ = "Cross-Lingual Sentiment Team"

from . import data_loader
from . import preprocess
from . import model
from . import train
from . import evaluate
from . import utils

__all__ = [
    "data_loader",
    "preprocess", 
    "model",
    "train",
    "evaluate",
    "utils",
]
