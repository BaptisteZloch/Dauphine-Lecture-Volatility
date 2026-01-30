import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

from investment_lab.data.option_db import OptionLoader
from investment_lab.data.rates_db import USRatesLoader
from investment_lab.dataclass import OptionLegSpec
from investment_lab.option_selection import select_options
from investment_lab.rates import compute_forward
from investment_lab.util import check_is_true


class OptionTradeABC(ABC):
    _REQUIRED_COLUMNS = ["date", "option_id", "expiration", "delta", "strike", "moneyness", "call_put", "spot", "ticker"]

    @classmethod
    def generate_trades(
        cls, start_date: datetime, end_date: datetime, tickers: list[str] | str, legs: list[OptionLegSpec], cost_neutral: bool = False
    ) -> pd.DataFrame:
        df_options = cls._load_option_data(start_date, end_date, process_kwargs={"ticker": tickers})
        df_trades = cls._select_options(df_options, legs, cost_neutral=cost_neutral).drop_duplicates(subset=["entry_date", "leg_name", "ticker"])
        df_trades_daily = cls._convert_trades_to_timeseries(df_trades)
        # merge back to get option data for the df_trades
        df_trades_daily = df_trades_daily.merge(df_options, on=["date", "option_id", "ticker"], how="left")
        df_trades_daily = df_trades_daily[df_trades_daily["date"].between(start_date, end_date)]
        df_trades_daily = df_trades_daily.drop_duplicates(subset=["date", "leg_name", "option_id"])
        df_trades_daily = cls._ffill_option_data(df_trades_daily)
        if "risk_free_rate" not in df_trades_daily.columns:
            start, end = df_trades_daily["date"].min(), df_trades_daily["date"].max()
            df_rates = USRatesLoader.load_data(start, end)
            df_trades_daily = compute_forward(df_options=df_trades_daily, df_rates=df_rates)

        return cls._delta_hedge(df_trades_daily)

    @classmethod
    def _load_option_data(cls, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """Concrete function that should be overridden in the child class. THis one wraps
            - `_load_data` to load the data from the source
            - `_preprocess_option_data` to preprocess the data after loading
        It performs also a check that the loaded data contains all required columns.

        Args:
            start_date (datetime): start date
            end_date (datetime): end date

        Returns:
            option data to be the input of the trade generation process.
        """
        logging.info("Loading option data from %s to %s", start_date, end_date)
        option_df = cls.load_data(start_date, end_date, **kwargs)
        missing_cols = set(cls._REQUIRED_COLUMNS).difference(option_df.columns)
        check_is_true(len(missing_cols) == 0, f"Option data is missing required columns: {missing_cols}")
        return cls._preprocess_option_data(option_df)

    @classmethod
    @abstractmethod
    def load_data(cls, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def _preprocess_option_data(cls, df_option: pd.DataFrame) -> pd.DataFrame:
        logging.info("Preprocessing option data.")
        return df_option

    @classmethod
    def _select_options(
        cls,
        df_options: pd.DataFrame,
        legs: list[OptionLegSpec],
        cost_neutral: bool = False,
    ) -> pd.DataFrame:
        df_list = []
        for leg in deepcopy(legs):
            leg_name = leg.pop("leg_name", "")
            weight = leg.pop("weight", np.nan)
            rebal_week_day = leg.pop("rebal_week_day", 1)
            check_is_true(np.all([0 <= rebal <= 4 for rebal in rebal_week_day]), "Error, provide a rebalance week day among {0,1,2,3,4}")
            logging.info("Selecting options for leg: %s using the rules:\n%s", leg_name, leg)
            selected_option_df = select_options(df_options, **leg)
            selected_option_df["leg_name"] = leg_name
            selected_option_df["weight"] = (weight / selected_option_df["spot"].where(selected_option_df["spot"] != 0, np.nan)).ffill()
            selected_option_df = selected_option_df[selected_option_df["date"].dt.day_of_week.isin(rebal_week_day)]
            df_list.append(selected_option_df.rename(columns={"date": "entry_date"}))

        df = pd.concat(df_list)
        if cost_neutral:
            df = cls._neutralize_cost(df)

        return df[["entry_date", "option_id", "expiration", "leg_name", "weight", "ticker"]]

    @classmethod
    def _neutralize_cost(cls, df_trades: pd.DataFrame) -> pd.DataFrame:
        logging.info("Adjusting weights to make the strategy cost neutral.")
        df_trades_cp = df_trades.copy()
        df_trades_cp["premium"] = df_trades_cp["weight"] * df_trades_cp["mid"]
        df_trades_cp["L/S"] = np.where(df_trades_cp["weight"] > 0, "Long", "Short")
        df_trade_pivot = df_trades_cp.pivot_table(index=["entry_date", "ticker"], columns="L/S", values="premium", aggfunc="sum")
        df_trade_pivot["missing_premium"] = -df_trade_pivot["Long"] - df_trade_pivot["Short"]
        df_trade_pivot["scaling_factor"] = np.where(
            df_trade_pivot["missing_premium"] < 0,
            (df_trade_pivot["Short"] + df_trade_pivot["missing_premium"]) / df_trade_pivot["Short"],
            (df_trade_pivot["Long"] + df_trade_pivot["missing_premium"]) / df_trade_pivot["Long"],
        )
        df_trades_cp = df_trades_cp.merge(
            df_trade_pivot.reset_index()[["entry_date", "ticker", "scaling_factor", "missing_premium"]],
            on=["entry_date", "ticker"],
            how="left",
        )
        df_trades_cp["weight"] = np.where(
            ((df_trades_cp["missing_premium"] < 0) & (df_trades_cp["weight"] < 0))
            | ((df_trades_cp["missing_premium"] > 0) & (df_trades_cp["weight"] > 0)),
            df_trades_cp["weight"] * df_trades_cp["scaling_factor"],
            df_trades_cp["weight"],
        )
        return df_trades_cp

    @classmethod
    def _convert_trades_to_timeseries(cls, df_trades: pd.DataFrame) -> pd.DataFrame:
        logging.info("Converting %s df_trades to daily time series", len(df_trades))
        df_trades_cp = df_trades.copy()
        df_trades_cp["date"] = df_trades_cp.apply(
            lambda r: pd.date_range(start=r["entry_date"], end=r["expiration"], freq="B"),
            axis=1,
        )
        df_trades_cp = df_trades_cp.explode("date").reset_index(drop=True)
        return df_trades_cp[["date", "option_id", "entry_date", "leg_name", "weight", "ticker"]]

    @classmethod
    def _ffill_option_data(cls, df_trades: pd.DataFrame) -> pd.DataFrame:
        logging.info("Forward filling option data for df_trades")
        return df_trades.sort_values(by=["option_id", "date"]).groupby("option_id", as_index=True, group_keys=False).apply(lambda x: x.ffill())

    @classmethod
    def _delta_hedge(cls, df_trades: pd.DataFrame) -> pd.DataFrame:
        """Delta hedge the trade previously generated.

        Args:
            df_trades: Trade DataFrame from `generate_trades`

        Returns:
            Same as input with additional row per date with the delta hedge position.
        """
        return df_trades


class OptionTrade(OptionLoader, OptionTradeABC):
    pass
