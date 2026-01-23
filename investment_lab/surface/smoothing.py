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
            beta: elasticity parameter (fixed)
            initial_guess: tuple of (alpha, rho, nu)

        Returns:
        Calibrated parameters (alpha, beta, rho, nu)
        """

        def objective(params: tuple[float, float, float, float]) -> float:
            self._params = params
            model_vol = self.transform(
                forward=forward, strike=strike, time_to_maturities=time_to_maturities
            )
            return mse(market_implied_vols, model_vol)

        bounds = [(1e-6, None), (0, 1), (-0.999, 0.999), (1e-6, None)]
        result = minimize(
            objective,
            self._params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "disp": False},
        )
        self._params = result.x
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
            time_to_maturities: time to maturity
            params: tuple of (alpha, beta, rho, nu)

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


class SVISmoother(VolSmoother):
    def __init__(
        self, initial_params: tuple[float, float, float, float, float]
    ) -> None:
        check_is_true(
            len(initial_params) == 5,
            "Initial parameters must be a tuple of (a, b, ρ, m, σ).",
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
            Calibrated parameters (a, b, ρ, m, σ)
        """

        def objective(params: tuple[float, float, float, float, float]) -> float:
            self._params = params
            model_vol = self.transform(
                forward=forward, strike=strike, time_to_maturities=time_to_maturities
            )
            return mse(market_implied_vols, model_vol)

        # Define bounds for parameters: a, b, ρ, m, σ
        bounds = [(1e-6, None), (0, None), (-0.999, 0.999), (None, None), (1e-6, None)]

        # Add constraints to ensure no arbitrage (variance must be positive)
        def no_arbitrage_constraint(
            params: tuple[float, float, float, float, float],
        ) -> float:
            a, b, rho, m, sigma = params
            forward = np.asarray(forward)
            strike = np.asarray(strike)
            forward_log_moneyness = np.log(strike / forward)
            total_variance = a + b * (
                rho * (forward_log_moneyness - m)
                + np.sqrt((forward_log_moneyness - m) ** 2 + sigma**2)
            )
            return np.min(total_variance)  # Ensure no negative variance

        result = minimize(
            objective,
            self._params,
            method="L-BFGS-B",
            bounds=bounds,
            constraints={"type": "ineq", "fun": no_arbitrage_constraint},
            options={"maxiter": 1000, "disp": False},
        )
        self._params = result.x
        return self

    def _transform(
        self,
        forward: pd.Series | np.ndarray,
        strike: pd.Series | np.ndarray,
        time_to_maturities: pd.Series | np.ndarray,
        **kwargs,
    ) -> pd.Series | np.ndarray:
        """
        Compute implied volatility using SVI model.
        Parameters:
            forward: forward price
            strike: strike price
            time_to_maturities: time to maturity
            params: tuple of (a, b, ρ, σ)
        Returns:
            SVI implied volatility
        """
        a, b, rho, m, sigma = self._params
        forward = np.asarray(forward)
        strike = np.asarray(strike)
        forward_log_moneyness = np.log(strike / forward)
        time_to_maturities = np.asarray(time_to_maturities)

        # SVI formula
        total_variance = a + b * (
            rho * (forward_log_moneyness - m)
            + np.sqrt((forward_log_moneyness - m) ** 2 + sigma**2)
        )
        total_variance = np.maximum(total_variance, 0)
        return np.sqrt(total_variance / time_to_maturities)
