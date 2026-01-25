import numpy as np
import pandas as pd


def mse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def sse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return np.sum((y_true - y_pred) ** 2)
