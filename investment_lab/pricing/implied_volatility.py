import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from investment_lab.pricing.black_scholes import (black_scholes_price,
                                                  vega_black_scholes)


def implied_volatility_vectorized(
    market_price: pd.Series,
    S: pd.Series,
    K: pd.Series,
    T: pd.Series,
    r: pd.Series,
    option_type: pd.Series,
    initial_guess: float = 0.2,
    tol: float = 1e-7,
    max_iterations: int = 10000,
) -> pd.Series:
    """
    Computes Implied Volatility using the Newton-Raphson method.
    Optimized for Pandas Series input.
    """
    # Initialize sigma with the initial guess, matching the index of the input
    sigma = pd.Series(initial_guess, index=market_price.index, dtype=float)
    logging.info("Calculate implied volatility using Newton-Raphson method")
    logging.info("Parameters: initial_guess=%s, tol=%s, max_iteration=%s", initial_guess, tol, max_iterations)
    for i in tqdm(range(max_iterations), desc="Calculating Implied Volatility", leave=True):
        # Calculate current prices and vega based on current sigma estimate
        current_price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega = vega_black_scholes(S, K, T, r, sigma)

        # Newton-Raphson step: sigma_new = sigma - f(sigma)/f'(sigma)
        # We add a small epsilon to vega to avoid division by zero
        price_diff = market_price - current_price

        # Only update sigma where vega is significant to avoid explosion
        sigma += (price_diff / vega.replace(0, np.nan)).fillna(0)

        # Optional: Keep sigma within realistic bounds (e.g., 0.001% to 500%)
        sigma = sigma.clip(lower=1e-5, upper=5.0)

        # Check for convergence: if the max difference is within tolerance, stop
        if price_diff.abs().max() < tol:
            logging.info("Converged after %s iterations", i + 1)
            break

    return sigma
