from typing import Optional

import numpy as np


def check_is_true(condition: bool, message: Optional[str] = None) -> None:
    if not condition:
        raise ValueError(message or "Condition is not true.")


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def sse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum((y_true - y_pred) ** 2)
