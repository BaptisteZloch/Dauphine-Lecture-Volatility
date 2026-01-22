from typing import Literal, Optional

import pandas as pd

from investment_lab.constants import TRADING_DAYS_PER_YEAR


def realized_volatility(returns: pd.Series) -> float:
    """
    Compute the annualized realized volatility of a daily returns series.
    Uses standard deviation of realized returns.

    Parameters:
        returns: Series of daily returns.

    Returns:
        Annualized realized volatility.
    """
    return returns.std() * (TRADING_DAYS_PER_YEAR**0.5)


def rolling_realized_volatility(
    returns: pd.Series,
    window: int,
    volatility_type: Literal["std"],
    volatility_kwargs: Optional[dict] = None,
) -> pd.Series:
    """
    Compute the rolling annualized realized volatility of daily a returns series.

    Parameters:
        returns: Series of daily returns.
        window: Rolling window size in number of periods.
        volatility_type: Type of volatility calculation ('std' for now).

    Returns:
        Series of rolling annualized realized volatility.
    """
    _VOLATILITY_TO_FUNC_MAP = {"std": realized_volatility}
    volatility_kwargs = volatility_kwargs or {}
    return returns.rolling(window).apply(
        _VOLATILITY_TO_FUNC_MAP[volatility_type], raw=False, **volatility_kwargs
    )
