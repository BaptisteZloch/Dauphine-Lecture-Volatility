from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


def black_scholes_price(
    S: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    K: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    T: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    r: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    sigma: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    option_type: np.ndarray[Any, np.dtype[np.str_]] | pd.Series | pd.Series,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series:
    """
    Args:
        S: underlying price
        K: strike price
        T: time to maturity (in years)
        r: risk-free rate
        sigma: implied volatility
        option_type: 'call' or 'put'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return np.where(
        option_type == "C",
        S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2),
        K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1),
    )


def black_scholes_greeks(
    S: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    K: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    T: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    r: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    sigma: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    option_type: np.ndarray[Any, np.dtype[np.str_]] | pd.Series | pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "delta": delta_black_scholes(S, K, T, r, sigma, option_type),
            "gamma": gamma_black_scholes(S, K, T, r, sigma),
            "vega": vega_black_scholes(S, K, T, r, sigma),
            "theta": theta_black_scholes(S, K, T, r, sigma, option_type),
            "rho": rho_black_scholes(S, K, T, r, sigma, option_type),
        }
    )


def delta_black_scholes(
    S: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    K: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    T: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    r: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    sigma: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    option_type: np.ndarray[Any, np.dtype[np.str_]] | pd.Series,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series:
    """
    Args:
        S: underlying price
        K: strike price
        T: time to maturity (in years)
        r: risk-free rate
        sigma: implied volatility
        option_type: 'call' or 'put'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    return np.where(
        option_type == "C",
        norm.cdf(d1),
        norm.cdf(d1) - 1,
    )


def vega_black_scholes(
    S: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    K: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    T: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    r: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    sigma: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series:
    """
    Args:
        S: underlying price
        K: strike price
        T: time to maturity (in years)
        r: risk-free rate
        sigma: implied volatility
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    return S * norm.pdf(d1) * np.sqrt(T)


def implied_volatility_black_scholes(
    market_price: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    S: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    K: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    T: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    r: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    option_type: np.ndarray[Any, np.dtype[np.str_]] | pd.Series,
    initial_guess: float = 0.2,
    tol: float = 1e-6,
    max_iterations: int = 100,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series:
    """
    Args:
        market_price: observed market price of the option
        S: underlying price
        K: strike price
        T: time to maturity (in years)
        r: risk-free rate
        option_type: 'call' or 'put'
        initial_guess: initial guess for volatility
        tol: tolerance for convergence
        max_iterations: maximum number of iterations
    """
    sigma = np.full_like(S, initial_guess)

    for _ in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega = vega_black_scholes(S, K, T, r, sigma)

        price_diff = market_price - price
        sigma += price_diff / vega

        if np.all(np.abs(price_diff) < tol):
            break

    return sigma


def theta_black_scholes(
    S: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    K: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    T: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    r: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    sigma: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    option_type: np.ndarray[Any, np.dtype[np.str_]] | pd.Series,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series:
    """
    Args:
        S: underlying price
        K: strike price
        T: time to maturity (in years)
        r: risk-free rate
        sigma: implied volatility
        option_type: 'call' or 'put'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2_call = r * K * np.exp(-r * T) * norm.cdf(d2)
    term2_put = r * K * np.exp(-r * T) * norm.cdf(-d2)

    return np.where(
        option_type == "C",
        term1 - term2_call,
        term1 + term2_put,
    )


def rho_black_scholes(
    S: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    K: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    T: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    r: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    sigma: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    option_type: np.ndarray[Any, np.dtype[np.str_]] | pd.Series,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series:
    """
    Args:
        S: underlying price
        K: strike price
        T: time to maturity (in years)
        r: risk-free rate
        sigma: implied volatility
        option_type: 'call' or 'put'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return np.where(
        option_type == "C",
        K * T * np.exp(-r * T) * norm.cdf(d2),
        -K * T * np.exp(-r * T) * norm.cdf(-d2),
    )


def gamma_black_scholes(
    S: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    K: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    T: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    r: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
    sigma: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series:
    """
    Args:
        S: underlying price
        K: strike price
        T: time to maturity (in years)
        r: risk-free rate
        sigma: implied volatility
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    return norm.pdf(d1) / (S * sigma * np.sqrt(T))
