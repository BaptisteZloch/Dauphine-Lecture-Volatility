import logging
from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from investment_lab.metrics.distance import mse
from investment_lab.surface.base import VolSmoother
from investment_lab.util import check_is_true


class SSVISmoother(VolSmoother):
    def __init__(self, initial_params: tuple[float, float, float, float]) -> None:
        check_is_true(
            len(initial_params) == 4,
            "Initial parameters must be a tuple of (sigma, rho, eta, lamb).",
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
        Calibrate SVI parameters to market implied volatilities.
        Parameters:
            forward: forward price
            strike: array of strikes
            time_to_maturities: time to maturity
            market_implied_vols: array of market implied volatilities

        Returns:
            Calibrated parameters (sigma, rho, eta, lamb)
        """

        def objective(params: tuple[float, float, float, float]) -> float:
            self._params = params
            model_total_variance = self.transform(
                forward=forward, strike=strike, time_to_maturities=time_to_maturities
            )
            # Calculate Mean Squared Error
            market_implied_variance = (market_implied_vols**2) * (time_to_maturities)
            return mse(market_implied_variance, model_total_variance)

        # Constraints for arbitrage-free SSVI:
        # |rho| < 1
        # eta > 0
        # lambda <= 0.5 (Necessary condition for no-butterfly arbitrage)
        bounds = [(0.00001, None), (-0.999, 0.999), (1e-6, 5.0), (1e-6, 0.5)]
        optimizer = "L-BFGS-B"
        logging.info("Fitting SSVI model on %s records", market_implied_vols.shape[0])
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
            "Successfully fitted SSVI Model. Found the following Parameters %s",
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
        Compute Total Variance w(k, theta) using SSVI with Power Law kernel.

        Parameters:
            params: tuple of (sigma, rho, eta, lamb)

        Returns:
            SSVI total variance
        """
        sigma, rho, eta, lamb = self._params
        # 1. Pre-calculate log-moneyness (k) and ATM Total Variance (theta) theta: ATM total variance (sigma_atm^2 * T)
        k = np.log(np.asarray(strike) / np.asarray(forward))
        theta = sigma * sigma * time_to_maturities
        # Power Law kernel: phi(theta)
        phi = eta / (theta**lamb)
        # SSVI Formula
        # w(k, theta) = 0.5 * theta * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2))
        inner_sqrt = np.sqrt((phi * k + rho) ** 2 + 1 - rho**2)
        return 0.5 * theta * (1 + rho * phi * k + inner_sqrt)
