"""
Utility functions for cross-lingual sentiment analysis.
"""

import os
import random
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

import yaml
import numpy as np
import torch


def setup_logging(
    level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format string for log messages
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Dictionary to save
        path: Output file path
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_device() -> torch.device:
    """
    Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device for computation
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")
    
    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with total and trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params
    }


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics dictionary as a readable string.
    
    Args:
        metrics: Dictionary of metric names and values
        precision: Decimal precision for floating point values
        
    Returns:
        Formatted string representation
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.{precision}f}")
        else:
            lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root
    """
    # Navigate up from src/ to project root
    current_file = Path(__file__).resolve()
    return current_file.parent.parent


class EarlyStopping:
    """
    Early stopping handler for training.
    """
    
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "max"
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' depending on metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        labels: Array of class labels
        
    Returns:
        Dictionary mapping class indices to weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )
    
    return {int(c): float(w) for c, w in zip(classes, weights)}


if __name__ == "__main__":
    # Test utilities
    logger = setup_logging()
    logger.info("Testing utilities module")
    
    config = load_config()
    logger.info(f"Loaded config with {len(config)} sections")
    
    set_seed(42)
    logger.info("Random seed set to 42")
    
    device = get_device()
    logger.info(f"Device: {device}")
