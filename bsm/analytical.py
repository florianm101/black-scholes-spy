"""Closed-form Black-Scholes-Merton pricing and analytical Greeks.

All functions accept scalars or NumPy arrays and broadcast accordingly.
Conventions:
    T      : time to expiry in years
    r, q   : continuously compounded risk-free rate and dividend yield
    sigma  : annualised volatility of log returns
    theta  : returned per YEAR (divide by 365 for the per-day figure traders quote)
    vega   : returned per UNIT of vol (divide by 100 for the per-vol-point figure)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["d1_d2", "bs_price", "bs_greeks"]

_VALID_KINDS = ("call", "put")


def _check_kind(kind: str) -> None:
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")


def d1_d2(S, K, T, r, sigma, q=0.0):
    """Return the Black-Scholes d1 and d2 terms."""
    S, K, T, r, sigma = map(np.asarray, (S, K, T, r, sigma))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    """Black-Scholes-Merton price of a European option on a continuous-yield asset.

    Parameters
    ----------
    S, K: float or array
        Spot and strike.
    T: float or array
        Time to expiry in years.
    r, q: float or array
        Risk-free rate and continuous dividend yield.
    sigma: float or array
        Annualised volatility.
    kind: {"call", "put"}

    Returns
    -------
    float or ndarray
    """
    _check_kind(kind)
    S, K, T, r, sigma = map(np.asarray, (S, K, T, r, sigma))
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    """Analytical first- and second-order Greeks.

    Returns
    -------
    dict with keys: delta, gamma, vega, theta, rho
        theta is per year; vega is per unit of vol.
    """
    _check_kind(kind)
    S, K, T, r, sigma = map(np.asarray, (S, K, T, r, sigma))
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    pdf = norm.pdf(d1)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)

    if kind == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -S * disc_q * pdf * sigma / (2 * np.sqrt(T))
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (
            -S * disc_q * pdf * sigma / (2 * np.sqrt(T))
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
        rho = -K * T * disc_r * norm.cdf(-d2)

    gamma = disc_q * pdf / (S * sigma * np.sqrt(T))
    vega = S * disc_q * pdf * np.sqrt(T)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
