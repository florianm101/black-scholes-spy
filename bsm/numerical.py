"""Numerical option pricing: binomial trees and Monte Carlo.

The closed form in `analytical.py` exists only for European vanillas under
geometric Brownian motion. These methods generalise: trees handle early
exercise, Monte Carlo handles path dependence and high dimensionality. The
European vanilla is the one contract all three price, which makes it the
natural regression test, if a tree or an MC estimator does not converge to
Black-Scholes here, it is wrong everywhere else too.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = ["binomial_price", "mc_price", "MCResult"]


class MCResult(NamedTuple):
    """Monte Carlo estimate with its uncertainty."""

    price: float
    stderr: float

    @property
    def ci95(self) -> tuple[float, float]:
        """95% confidence interval for the price estimate."""
        return (self.price - 1.96 * self.stderr, self.price + 1.96 * self.stderr)


def binomial_price(S, K, T, r, sigma, q=0.0, kind="call", steps=500, american=False):
    """Cox-Ross-Rubinstein binomial tree with vectorised backward induction.

    Up/down factors u = exp(sigma*sqrt(dt)), d = 1/u, with risk-neutral
    probability p = (exp((r-q)*dt) - d) / (u - d). Discretisation error is
    O(1/n) with a characteristic odd/even oscillation as the strike falls
    between tree nodes.

    Parameters
    ----------
    steps: int
        Number of time steps. Error ~ O(1/steps).
    american: bool
        If True, allow early exercise (take max with intrinsic at each node).

    Returns
    -------
    float
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"risk-neutral probability p={p:.4f} outside [0,1]; "
            "increase steps or check inputs (dt too large relative to sigma)"
        )
    disc = np.exp(-r * dt)

    j = np.arange(steps + 1)
    ST = S * u**j * d ** (steps - j)
    vals = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)

    for step in range(steps - 1, -1, -1):
        vals = disc * (p * vals[1:] + (1 - p) * vals[:-1])
        if american:
            j = np.arange(step + 1)
            S_node = S * u**j * d ** (step - j)
            intrinsic = (
                np.maximum(S_node - K, 0.0) if kind == "call" else np.maximum(K - S_node, 0.0)
            )
            vals = np.maximum(vals, intrinsic)

    return float(vals[0])


def mc_price(
    S, K, T, r, sigma, q=0.0, kind="call", n_paths=100_000, antithetic=True, seed=None
):
    """Terminal-value Monte Carlo under GBM.

    Simulates S_T = S*exp((r - q - sigma^2/2)*T + sigma*sqrt(T)*Z) directly --
    no time stepping is needed because the payoff depends only on the terminal
    value. Standard error shrinks as O(1/sqrt(N)).

    Antithetic variates: each draw Z is paired with -Z and the pair averaged.
    The two payoffs are negatively correlated, so the pair-average has lower
    variance than two independent samples. Note the effective sample size is
    then n_paths//2 pairs, which is what the reported standard error uses.

    Parameters
    ----------
    n_paths: int
        Total number of normal draws. With antithetic=True this is split into
        n_paths//2 antithetic pairs.
    antithetic: bool
        Enable antithetic variate variance reduction.
    seed: int or None
        Seed for the random generator (reproducibility).

    Returns
    -------
    MCResult
        Named tuple of (price, stderr) with a .ci95 property.
    """
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)
    drift = (r - q - 0.5 * sigma**2) * T
    vol_term = sigma * np.sqrt(T)

    if antithetic:
        n = n_paths // 2
        if n < 1:
            raise ValueError("n_paths must be >= 2 when antithetic=True")
        z = rng.standard_normal(n)
        z = np.concatenate([z, -z])
    else:
        n = n_paths
        z = rng.standard_normal(n_paths)

    ST = S * np.exp(drift + vol_term * z)
    payoff = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)

    if antithetic:
        # Each antithetic pair is one iid observation.
        sample = disc * 0.5 * (payoff[:n] + payoff[n:])
    else:
        sample = disc * payoff

    return MCResult(float(sample.mean()), float(sample.std(ddof=1) / np.sqrt(n)))
