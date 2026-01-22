import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from investment_lab.util import check_is_true


class DataLoader(ABC):
    _EXTENSION_TO_LOADER = {
        "parquet": pd.read_parquet,
        "csv": pd.read_csv,
        "xlsx": pd.read_excel,
    }

    @classmethod
    def load_data(
        cls,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        load_kwargs: Optional[dict] = None,
        process_kwargs: Optional[dict] = None,
        extra_fields_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        file_path = cls._get_path()
        min_date, max_date = cls._get_valid_date_range()
        start_date = start_date or min_date
        end_date = end_date or max_date
        check_is_true(start_date <= end_date, "start_date must be before end_date")
        check_is_true(
            start_date >= min_date and end_date <= max_date,
            f"Data is only available between {min_date.date()} and {max_date.date()}",
        )
        extension = file_path.split(".")[-1]
        check_is_true(
            extension in cls._EXTENSION_TO_LOADER,
            f"Unsupported file extension: {extension}",
        )
        logging.info("Reading between %s %s from %s", start_date, end_date, file_path)
        df = cls._EXTENSION_TO_LOADER[extension](file_path, **(load_kwargs or {}))
        df["date"] = df["date"].astype("datetime64[ns]")
        df_processed = cls._add_extra_fields(
            cls._process_loaded_data(df, **(process_kwargs or {})),
            **(extra_fields_kwargs or {}),
        )
        return df_processed[df_processed["date"].between(start_date, end_date)]

    @classmethod
    @abstractmethod
    def _get_path(cls) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _get_valid_date_range(cls) -> tuple[datetime, datetime]:
        raise NotImplementedError

    @classmethod
    def _process_loaded_data(cls, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df

    @classmethod
    def _add_extra_fields(cls, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df
