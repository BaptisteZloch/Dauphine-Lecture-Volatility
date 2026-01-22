from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from investment_lab.data.data_loader import DataLoader

ROOT_PATH = r"../"


class OptionLoader(DataLoader):
    @classmethod
    def _get_path(cls) -> str:
        return rf"{ROOT_PATH}/data/optiondb_2016_2023.parquet"

    @classmethod
    def _get_valid_date_range(cls) -> tuple[datetime, datetime]:
        return (datetime(2016, 1, 2), datetime(2023, 12, 30))

    @classmethod
    def _process_loaded_data(
        cls, df: pd.DataFrame, *, ticker: str | Sequence[str], **kwargs
    ) -> pd.DataFrame:
        if isinstance(ticker, str):
            ticker = [ticker]
        else:
            ticker = [t for t in ticker]
        df = df[df["ticker"].isin(ticker)]
        df["volume"] = df["volume"].fillna(0)
        return cls._compute_final_payoff(df)

    @classmethod
    def _add_extra_fields(cls, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df["day_to_expiration"] = (df["expiration"] - df["date"]).dt.days
        df["moneyness"] = df["strike"] / df["spot"]
        return df

    @staticmethod
    def _compute_final_payoff(df_option: pd.DataFrame) -> pd.DataFrame:
        expiring_filter = df_option["date"] == df_option["expiration"].copy()
        expiring_calls_filter = expiring_filter & (df_option["call_put"] == "C")
        expiring_puts_filter = expiring_filter & (df_option["call_put"] == "P")

        call_payoff = (df_option["spot"] - df_option["strike"]).clip(lower=0)
        put_payoff = (df_option["strike"] - df_option["spot"]).clip(lower=0)

        for c in ("mid", "bid", "ask"):
            df_option[c] = np.where(
                expiring_calls_filter,
                call_payoff,
                np.where(expiring_puts_filter, put_payoff, df_option[c]),
            )
        return df_option


class SPYOptionLoader(OptionLoader):
    @classmethod
    def _get_path(cls) -> str:
        return rf"{ROOT_PATH}/data/spy_2020_2022.parquet"

    @classmethod
    def _get_valid_date_range(cls) -> tuple[datetime, datetime]:
        return (datetime(2020, 1, 2), datetime(2022, 12, 30))


class AAPLOptionLoader(OptionLoader):
    @classmethod
    def _get_path(cls) -> str:
        return rf"{ROOT_PATH}/data/aapl_2016_2023.parquet"

    @classmethod
    def _get_valid_date_range(cls) -> tuple[datetime, datetime]:
        return (datetime(2016, 1, 2), datetime(2023, 12, 31))


def extract_spot_from_options(df_options: pd.DataFrame) -> pd.DataFrame:
    return (
        df_options[["date", "spot"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
