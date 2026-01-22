from typing import Literal

import pandas as pd

from investment_lab.util import check_is_true


def select_options(
    df_options: pd.DataFrame,
    call_or_put: Literal["C", "P"],
    strike_col: Literal["delta", "strike", "moneyness"],
    strike_target: float,
    day_to_expiry_target: int,
) -> pd.DataFrame:
    """Find the closest strike and maturity to a target strike and target expiration (in days).
    THe selection is done by ticker and date.

    Args:
        df_options: cross section of daily options.
        call_or_put: Call or put flag (C or P) to filter the input dataset.
        strike_col: The name of the strike column among moneyness, delta, strike.
        strike_target (float): The target delta to select for each day and each ticker.
        day_to_expiry_target (int): The target expiration in days.

    Returns:
        pd.DataFrame: The selected options.
    """
    return select_closest_strike(
        select_closest_maturity(df_options, target=day_to_expiry_target),
        strike_col=strike_col,
        target=strike_target,
        call_put=call_or_put,
    ).drop_duplicates(["date", "ticker"])


def select_closest_maturity(df_option: pd.DataFrame, target: int) -> pd.DataFrame:
    return _select_close_target(df_option, target_col="day_to_expiration", target=target, groupby_cols=["date", "ticker"])


def select_closest_strike(
    df_option: pd.DataFrame, strike_col: Literal["delta", "strike", "moneyness"], target: float, call_put: Literal["C", "P"] = "C"
) -> pd.DataFrame:
    check_is_true(call_put in ["C", "P"], "Error, call_put must be either P or C.")
    df_option_filtered = df_option[df_option["call_put"] == call_put].copy()
    return _select_close_target(df_option_filtered, target_col=strike_col, target=target, groupby_cols=["date", "ticker", "expiration"])


def _select_close_target(df: pd.DataFrame, target_col: str, target: float | int, groupby_cols: list[str]) -> pd.DataFrame:
    """Find the set option defined by groupby_cols closest to the target in target_col.
    ie: minimize the absolute distance in each group.

    Args:
        df (pd.DataFrame): Cross-section of daily options.
        target_col (str): Name of the column to minimize the absolute distance to target.
        target (float | int): Target to minimize the distance to.
        groupby_cols (Sequence[str]): Set of column defining a group.

    Returns:
        pd.DataFrame: Subset of df with only the options closest to the target in each group.
    """
    missing_cols = set(groupby_cols + [target_col]) - set(df.columns)
    check_is_true(len(missing_cols) == 0, f"Error, the following columns {missing_cols} are missingfrom the input dataframe")
    df["criterion"] = abs(df[target_col] - target)
    df_minimum_dist = df.groupby(groupby_cols)[["criterion"]].min().reset_index()
    return df.merge(df_minimum_dist, on=[*groupby_cols, "criterion"], how="inner").drop(columns=["criterion"])
