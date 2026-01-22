from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd

from investment_lab.util import check_is_true


class VolSmoother(ABC):
    def __init__(self, initial_params: tuple):
        self._params = initial_params

    @property
    def params(self) -> tuple:
        return self._params

    def fit_transform(
        self,
        forward: float | pd.Series | np.ndarray,
        strike: pd.Series | np.ndarray,
        time_to_maturities: pd.Series | np.ndarray | float,
        market_implied_vols: pd.Series | np.ndarray,
        **kwargs,
    ) -> pd.Series | np.ndarray:
        return self.fit(
            forward=forward,
            strike=strike,
            time_to_maturities=time_to_maturities,
            market_implied_vols=market_implied_vols,
            **kwargs,
        ).transform(
            forward=forward,
            strike=strike,
            time_to_maturities=time_to_maturities,
            **kwargs,
        )

    def fit(
        self,
        forward: float | pd.Series | np.ndarray,
        strike: pd.Series | np.ndarray,
        time_to_maturities: pd.Series | np.ndarray | float,
        market_implied_vols: pd.Series | np.ndarray,
        **kwargs,
    ) -> Self:
        forward = np.asarray(forward)
        time_to_maturities = np.asarray(time_to_maturities)
        check_is_true(
            market_implied_vols.shape == strike.shape,
            "market_implied_vols and strike must have the same shape.",
        )
        check_is_true(
            np.isnan(forward).sum() == 0
            and np.isnan(time_to_maturities).sum() == 0
            and np.isnan(strike).sum() == 0,
            "Forward, time to maturities, and strike must not contain NaN values.",
        )
        return self._fit(
            forward=forward,
            strike=strike,
            time_to_maturities=time_to_maturities,
            market_implied_vols=market_implied_vols,
            **kwargs,
        )

    @abstractmethod
    def _fit(
        self,
        forward: pd.Series | np.ndarray,
        strike: pd.Series | np.ndarray,
        time_to_maturities: pd.Series | np.ndarray,
        market_implied_vols: pd.Series | np.ndarray,
        **kwargs,
    ) -> Self:
        raise NotImplementedError

    def transform(
        self,
        forward: float | pd.Series | np.ndarray,
        strike: pd.Series | np.ndarray,
        time_to_maturities: pd.Series | np.ndarray | float,
        **kwargs,
    ) -> pd.Series | np.ndarray:
        forward = np.asarray(forward)
        time_to_maturities = np.asarray(time_to_maturities)
        return self._transform(
            forward=forward,
            strike=strike,
            time_to_maturities=time_to_maturities,
            **kwargs,
        )

    @abstractmethod
    def _transform(
        self,
        forward: pd.Series | np.ndarray,
        strike: pd.Series | np.ndarray,
        time_to_maturities: pd.Series | np.ndarray,
        **kwargs,
    ) -> pd.Series | np.ndarray:
        raise NotImplementedError
