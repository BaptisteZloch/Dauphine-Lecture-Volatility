import logging
from datetime import datetime
from typing import Self

import pandas as pd
from tqdm import tqdm

from investment_lab.util import check_is_true


class StrategyBacktester:
    _BACKTEST_COLS = [
        "date",
        "option_id",
        "weight",
        "mid",
        "entry_date",
        "expiration",
        "spot",
        "implied_volatility",
        "risk_free_rate",
        "delta",
        "theta",
        "vega",
        "gamma",
        "rho",
    ]
    _PNL_COLS = [
        "pnl",
        "delta_pnl",
        "gamma_pnl",
        "theta_pnl",
        "vega_pnl",
        "rho_pnl",
        "residual_pnl",
        "leverage",
        "cashflow",
    ]

    def __init__(self, df_positions: pd.DataFrame) -> None:
        missing_cols = set(self._BACKTEST_COLS).difference(df_positions.columns)
        check_is_true(
            len(missing_cols) == 0,
            f"Positions data is missing required columns: {missing_cols}",
        )
        check_is_true(
            len(df_positions) > 2,
            "Positions data is empty or too small to run backtest.",
        )
        self._df_positions = df_positions
        self._is_backtested = False
        self._df_pnl = pd.DataFrame()
        self._df_nav = pd.DataFrame()
        self._df_metainfo = pd.DataFrame()
        self._df_drifted_positions = pd.DataFrame()

    @property
    def pnl(self) -> pd.DataFrame:
        check_is_true(
            self._is_backtested,
            "Backtest has not been run yet. Call 'compute_backtest' method first.",
        )
        return self._df_pnl

    @property
    def nav(self) -> pd.DataFrame:
        check_is_true(
            self._is_backtested,
            "Backtest has not been run yet. Call 'compute_backtest' method first.",
        )
        return self._df_nav

    @property
    def metainfo(self) -> pd.DataFrame:
        check_is_true(
            self._is_backtested,
            "Backtest has not been run yet. Call 'compute_backtest' method first.",
        )
        return self._df_metainfo

    @property
    def drifted_positions(self) -> pd.DataFrame:
        check_is_true(
            self._is_backtested,
            "Backtest has not been run yet. Call 'compute_backtest' method first.",
        )
        return self._df_drifted_positions

    def compute_backtest(self) -> Self:
        df_positions = self._df_positions[self._BACKTEST_COLS].sort_values(["option_id", "date"]).copy()
        logging.info("Computing period to period difference.")
        df_positions["dv"] = df_positions.groupby(["option_id"])["mid"].diff().fillna(0)
        df_positions["dr"] = df_positions.groupby(["option_id"])["risk_free_rate"].diff().fillna(0)
        df_positions["dsigma"] = df_positions.groupby(["option_id"])["implied_volatility"].diff().fillna(0)
        df_positions["dS"] = df_positions.groupby(["option_id"])["spot"].diff().fillna(0)
        df_positions["dt"] = 1

        df_positions["prev_theta"] = df_positions.groupby("option_id")["theta"].shift(1).fillna(method="bfill")
        df_positions["prev_gamma"] = df_positions.groupby("option_id")["gamma"].shift(1).fillna(method="bfill")
        df_positions["prev_delta"] = df_positions.groupby("option_id")["delta"].shift(1).fillna(method="bfill")
        df_positions["prev_vega"] = df_positions.groupby("option_id")["vega"].shift(1).fillna(method="bfill")
        df_positions["prev_rho"] = df_positions.groupby("option_id")["rho"].shift(1).fillna(method="bfill")

        df_positions["obs_date"] = df_positions["date"].apply(lambda x: x - pd.Timedelta(days=1))
        df_pnl = pd.DataFrame(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0]],
            columns=[
                "pnl",
                "delta_pnl",
                "gamma_pnl",
                "theta_pnl",
                "vega_pnl",
                "rho_pnl",
                "residual_pnl",
                "leverage",
                "cashflow",
            ],
            index=[df_positions["date"].min() - pd.Timedelta(days=1)],
        )
        df_nav = pd.DataFrame(
            [[1]],
            columns=["NAV"],
            index=[df_positions["date"].min() - pd.Timedelta(days=1)],
        )
        logging.info(
            "Starting backtest computation over %s unique dates.",
            len(df_positions["date"].unique()),
        )
        drifted_positions = []
        for d in tqdm(df_positions["date"].sort_values().unique()):
            df_day = df_positions[df_positions["date"] == d].copy()
            df_day = df_day.merge(df_nav, left_on="obs_date", right_index=True, how="left")
            df_day["scaled_weight"] = df_day["weight"] * df_day["NAV"]

            df_day["pnl"] = df_day["scaled_weight"] * df_day["dv"]
            df_day["gamma_pnl"] = 0.5 * df_day["scaled_weight"] * df_day["dS"] ** 2 * df_day["prev_gamma"]
            df_day["delta_pnl"] = df_day["scaled_weight"] * df_day["dS"] * df_day["prev_delta"]
            df_day["theta_pnl"] = df_day["scaled_weight"] * df_day["dt"] * df_day["prev_theta"]
            df_day["vega_pnl"] = df_day["scaled_weight"] * df_day["dsigma"] * df_day["prev_vega"]
            df_day["rho_pnl"] = df_day["scaled_weight"] * df_day["dr"] * df_day["prev_rho"]
            df_day["residual_pnl"] = (
                df_day["pnl"] - df_day["delta_pnl"] - df_day["gamma_pnl"] - df_day["theta_pnl"] - df_day["vega_pnl"] - df_day["rho_pnl"]
            )
            df_day["leverage"] = df_day["scaled_weight"] * df_day["spot"]
            df_day["cashflow"] = 0
            df_day.loc[df_day["entry_date"] == df_day["date"], "cashflow"] = -df_day["scaled_weight"] * df_day["mid"]
            df_day.loc[df_day["expiration"] == df_day["date"], "cashflow"] = df_day["scaled_weight"] * df_day["mid"]

            df_pnl = pd.concat([df_pnl, df_day.groupby("date")[self._PNL_COLS].sum()])
            if d not in df_nav.index:
                latest_nav = df_nav[df_nav.index == df_nav.index.max()].iloc[0]
            else:
                latest_nav = df_nav.loc[d]
            df_nav.loc[d] = latest_nav + df_pnl.loc[d, "pnl"]
            drifted_positions.append(df_day)

        logging.info("Backtest computation completed.")
        self._is_backtested = True
        self._df_pnl = df_pnl.drop(columns=["leverage", "cashflow"]).copy()
        self._df_nav = df_nav.copy()
        self._df_metainfo = df_pnl[["leverage", "cashflow"]].copy()
        self._df_drifted_positions = pd.concat(drifted_positions).reset_index(drop=True)
        return self

    def __del__(self):
        logging.info("Deleting StrategyBacktest instance and freeing up memory.")
        self._df_positions = pd.DataFrame()
        self._df_pnl = pd.DataFrame()
        self._df_nav = pd.DataFrame()
        self._df_metainfo = pd.DataFrame()
        self._df_drifted_positions = pd.DataFrame()
