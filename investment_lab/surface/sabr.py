import logging
from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from investment_lab.metrics.distance import mse
from investment_lab.surface.base import VolSmoother
from investment_lab.util import check_is_true


class SABRSmoother(VolSmoother):
    def __init__(self, initial_params: tuple[float, float, float, float]) -> None:
        check_is_true(
            len(initial_params) == 4,
            "Initial parameters must be a tuple of (alpha, beta, rho, nu).",
        )
        super().__init__(initial_params)

    def _fit(
        self,
        forward: float | pd.Series | np.ndarray,
        strike: pd.Series | np.ndarray,
        time_to_maturities: pd.Series | np.ndarray | float,
        market_implied_vols: pd.Series | np.ndarray,
        **kwargs,
    ) -> Self:
        """
        Calibrate SABR parameters to market implied volatilities.

        Parameters:
            forward: forward price
            strike: array of strikes
            time_to_maturities: time to maturity
            market_implied_vols: array of market implied volatilities

        Returns:
            The fitted class.
        """

        def objective(params: tuple[float, float, float, float]) -> float:
            self._params = params
            model_vol = self.transform(
                forward=forward, strike=strike, time_to_maturities=time_to_maturities
            )
            return mse(market_implied_vols, model_vol)

        bounds = [(1e-6, None), (0, 1), (-0.999, 0.999), (1e-6, None)]
        optimizer = "L-BFGS-B"
        logging.info("Fitting SABR model on %s records", market_implied_vols.shape[0])
        logging.info("Initial guess: %s", self._params)
        logging.info("Parameter bounds: %s", bounds)
        logging.info("Solver: %s", optimizer)
        result = minimize(
            objective,
            self._params,
            method=optimizer,
            bounds=bounds,
            options={"maxiter": 1000, "disp": True},
        )
        self._params = result.x
        logging.info(
            "Successfully fitted SABR Model. Found the following Parameters %s",
            self._params,
        )
        return self

    def _transform(
        self,
        forward: pd.Series | np.ndarray,
        strike: pd.Series | np.ndarray,
        time_to_maturities: pd.Series | np.ndarray,
        **kwargs,
    ) -> pd.Series | np.ndarray:
        """
        Compute implied volatility using SABR model.

        Parameters:
            forward: forward price
            strike: strike price
            time_to_maturities: time to maturity (in years)

        Returns:
            SABR implied volatility
        """
        alpha, beta, rho, nu = self._params
        forward = np.asarray(forward)
        time_to_maturities = np.asarray(time_to_maturities)

        z = (
            (nu / alpha)
            * (forward * strike) ** ((1 - beta) / 2)
            * np.log(forward / strike)
        )
        chi = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

        num = alpha
        denom = (forward * strike) ** ((1 - beta) / 2) * (
            1
            + ((1 - beta) ** 2 / 24) * (np.log(forward / strike)) ** 2
            + ((1 - beta) ** 4 / 1920) * (np.log(forward / strike)) ** 4
        )

        vol = (num / denom) * (z / chi)
        sabr_implied_vol = vol * np.sqrt(
            1
            + ((1 - beta) ** 2 / 24)
            * (alpha / ((forward * strike) ** ((1 - beta) / 2))) ** 2
            * time_to_maturities
            + (rho * beta * nu * alpha / ((forward * strike) ** ((1 - beta) / 2)))
            * time_to_maturities
            + ((2 - 3 * rho**2) / 24) * (nu**2) * time_to_maturities
        )
        return np.where(
            forward == strike, alpha / (forward ** (1 - beta)), sabr_implied_vol
        )
