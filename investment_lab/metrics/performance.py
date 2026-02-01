import pandas as pd

from investment_lab.constants import TRADING_DAYS_PER_YEAR
from investment_lab.metrics.util import returns_to_levels
from investment_lab.metrics.volatility import realized_volatility


def realized_returns(returns: pd.Series) -> float:
    """Compute annualized realized return

    Args:
        returns (pd.Series): _description_

    Returns:
        float: Annualized realized returns.
    """
    return returns.mean() * TRADING_DAYS_PER_YEAR


def sharpe_ratio(returns: pd.Series, risk_free_rate: float | pd.Series = 0.0) -> float:
    """Compute the annualized Sharpe ratio of a daily returns series.

    Parameters:
        returns: Series of daily returns.
        risk_free_rate: Annualized risk free rate or daily risk free rate series.

    Returns:
        Annualized Sharpe ratio.
    """
    return (
        realized_returns(excess_return(returns, risk_free_rate))
    ) / realized_volatility(returns)


def excess_return(
    returns: pd.Series,
    risk_free_rate: float | pd.Series = 0.0,
    annualized_risk_free_rate: bool = True,
) -> pd.Series:
    """Compute the annualized Sharpe ratio of a daily returns series.

    Parameters:
        returns: Series of daily returns.
        risk_free_rate: Annualized risk free rate or daily risk free rate series.

    Returns:
        Annualized Sharpe ratio.
    """
    return returns - (
        risk_free_rate
        if annualized_risk_free_rate
        else annualized_risk_free_rate / TRADING_DAYS_PER_YEAR
    )


def drawdown(returns: pd.Series) -> pd.Series:
    """Compute the drawdown series from a daily returns series.

    Parameters:
        returns: Series of daily returns.

    Returns:
        Series of drawdowns.
    """
    nav = returns_to_levels(returns)
    return (nav / nav.cummax()) - 1


def max_drawdown(returns: pd.Series) -> float:
    """Compute the maximum drawdown from a daily returns series.

    Parameters:
        returns: Series of daily returns.
    Returns:
        maximum drawdown of the returns series.
    """
    return drawdown(returns).min()


def calmar_ratio(returns: pd.Series) -> float:
    """Compute the Calmar ratio from a daily returns series.

    Parameters:
        returns: Series of daily returns.

    Returns:
        Calmar ratio of the returns series.
    """
    annualized_return = realized_returns(returns)
    maximum_drawdown = -max_drawdown(returns)
    if maximum_drawdown == 0:
        return float("inf")
    return annualized_return / maximum_drawdown
