"""Implied volatility inversion.

There is no closed-form inverse of the Black-Scholes formula in sigma, so we
root-find. The strategy is Newton-Raphson (quadratic convergence, uses vega as
the analytical derivative) with a bisection fallback (linear convergence but
cannot fail on a bracketed root).

Known limitation, by design rather than by bug: where vega is numerically zero;
deep in-the-money, short-dated options whose price equals discounted
intrinsic to machine precision. In these cases, the price carries NO information about
volatility and no solver can recover it. Any sigma reprices such an option
within tolerance. This is why practitioners build volatility surfaces from
out-of-the-money quotes only, where the entire premium is time value.
"""

from __future__ import annotations

import numpy as np

from .analytical import bs_greeks, bs_price

__all__ = ["implied_vol", "no_arbitrage_bounds"]


def no_arbitrage_bounds(S, K, T, r, q=0.0, kind="call"):
    """Return (lower, upper) no-arbitrage bounds on a European option price."""
    fwd = S * np.exp(-q * T)
    disc_k = K * np.exp(-r * T)
    if kind == "call":
        return max(fwd - disc_k, 0.0), fwd
    return max(disc_k - fwd, 0.0), disc_k

def is_identifiable(sigma, S, K, T, r, q, kind, min_price_move=1e-8):
    """True if a one-vol-point move in sigma changes the price measurably.

    Where it is False the price is pinned to intrinsic, every candidate sigma
    reprices the option within tolerance, and implied vol is not identifiable.
    """
    vega = float(bs_greeks(S, K, T, r, sigma, q, kind)["vega"])
    return vega * 0.01 > min_price_move

def implied_vol(
    price,
    S,
    K,
    T,
    r,
    q=0.0,
    kind="call",
    tol=1e-8,
    max_iter=100,
    sigma_init=0.3,
    sigma_bounds=(1e-6, 5.0),
    verbose=False,
):
    """Back out Black-Scholes implied volatility from an option price.

    Parameters
    ----------
    price: float
        Observed (market) option price.
    S, K, T, r, q: float
        Standard Black-Scholes inputs.
    kind: {"call", "put"}
    tol: float
        Convergence tolerance on sigma.
    sigma_init: float
        Newton starting point. 0.3 is a reasonable default for equity index options.
    sigma_bounds: tuple
        Bracket used by the bisection fallback.
    verbose: bool
        Print the Newton iteration trace.

    Returns
    -------
    float
        Implied volatility, or np.nan if the price violates no-arbitrage bounds
        or the root cannot be bracketed or the price is insensitive to volatility across
        the whole bracket.
    """
    lower, _ = no_arbitrage_bounds(S, K, T, r, q, kind)
    if price < lower - 1e-10:
        return np.nan

    # Newton-Raphson
    sigma = float(sigma_init)
    for i in range(max_iter):
        p = float(bs_price(S, K, T, r, sigma, q, kind))
        v = float(bs_greeks(S, K, T, r, sigma, q, kind)["vega"])
        if v < 1e-10:
            break  # objective is flat; Newton is useless here
        step = (p - price) / v
        sigma -= step
        if verbose:
            print(f"  NR iter {i}: sigma={sigma:.8f}  price_err={p - price:+.3e}")
        if sigma <= 0:
            break
        if abs(step) < tol:
            if not is_identifiable(sigma, S, K, T, r, q, kind):
                return np.nan
            return float(sigma)

    # Bisection fallback
    lo, hi = sigma_bounds
    p_lo = float(bs_price(S, K, T, r, lo, q, kind))
    p_hi = float(bs_price(S, K, T, r, hi, q, kind))
    if not (p_lo - 1e-9 <= price <= p_hi + 1e-9):
        return np.nan 
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(bs_price(S, K, T, r, mid, q, kind)) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    sigma = 0.5 * (lo + hi)
    if not is_identifiable(sigma, S, K, T, r, q, kind):
        return np.nan
    return float(sigma)