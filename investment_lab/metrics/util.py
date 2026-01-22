import pandas as pd


def levels_to_returns(levels: pd.Series) -> pd.Series:
    """Convert a series of price levels to returns.
    Process wide format only.

    Parameters:
        levels: Series of price levels.

    Returns:
        Series of returns.
    """
    return levels.pct_change()


def returns_to_levels(returns: pd.Series) -> pd.Series:
    """Convert a series of returns to price levels.
    Process wide format only.

    Parameters:
        returns: Series of returns.

    Returns:
        Series of price levels.
    """
    return (1 + returns).cumprod()
