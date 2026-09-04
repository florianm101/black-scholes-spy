"""Tests that the numerical methods converge to the closed form.

If a tree or Monte Carlo estimator cannot reproduce Black-Scholes on a
European vanilla -- the one contract where an exact answer exists -- it cannot
be trusted on the exotics where no exact answer does.
"""

import numpy as np
import pytest

from bsm import binomial_price, bs_price, mc_price

PARAMS = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_binomial_converges_to_closed_form(kind):
    exact = float(bs_price(**PARAMS, kind=kind))
    approx = binomial_price(**PARAMS, kind=kind, steps=2000)
    assert approx == pytest.approx(exact, abs=1e-3)


def test_binomial_error_decreases_with_steps():
    """Error should fall roughly as O(1/n). Compare coarse vs fine, averaging
    over neighbouring step counts to smooth the known odd/even oscillation."""
    exact = float(bs_price(**PARAMS))

    def avg_err(n):
        return np.mean([abs(binomial_price(**PARAMS, steps=n + k) - exact) for k in range(4)])

    assert avg_err(1000) < avg_err(50)


@pytest.mark.parametrize("K", [80, 100, 125])
def test_binomial_matches_across_moneyness(K):
    p = dict(PARAMS, K=K)
    assert binomial_price(**p, steps=3000) == pytest.approx(float(bs_price(**p)), abs=2e-3)


def test_american_call_equals_european_without_dividends():
    """Classic result: it is never optimal to exercise an American call early
    on a non-dividend-paying underlying, so the two prices coincide."""
    eu = binomial_price(**PARAMS, kind="call", steps=800, american=False)
    am = binomial_price(**PARAMS, kind="call", steps=800, american=True)
    assert am == pytest.approx(eu, abs=1e-8)


def test_american_put_carries_early_exercise_premium():
    """An American put IS worth more than its European counterpart."""
    eu = binomial_price(**PARAMS, kind="put", steps=800, american=False)
    am = binomial_price(**PARAMS, kind="put", steps=800, american=True)
    assert am > eu + 1e-4


def test_binomial_rejects_invalid_step_count():
    with pytest.raises(ValueError):
        binomial_price(**PARAMS, steps=0)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_monte_carlo_within_confidence_interval(kind):
    exact = float(bs_price(**PARAMS, kind=kind))
    res = mc_price(**PARAMS, kind=kind, n_paths=400_000, seed=12345)
    lo, hi = res.ci95
    assert lo <= exact <= hi, f"exact {exact} outside CI ({lo}, {hi})"


def test_antithetic_variates_reduce_standard_error():
    """The core claim of the variance-reduction technique, tested directly at
    equal path counts across several seeds."""
    ratios = []
    for seed in range(6):
        plain = mc_price(**PARAMS, n_paths=100_000, antithetic=False, seed=seed)
        anti = mc_price(**PARAMS, n_paths=100_000, antithetic=True, seed=seed)
        ratios.append(plain.stderr / anti.stderr)
    assert np.mean(ratios) > 1.1, f"no variance reduction observed: {np.mean(ratios):.3f}"


def test_monte_carlo_standard_error_scales_as_inverse_sqrt_n():
    """A 100x increase in paths should cut the SE by roughly 10x."""
    small = mc_price(**PARAMS, n_paths=10_000, antithetic=False, seed=7)
    large = mc_price(**PARAMS, n_paths=1_000_000, antithetic=False, seed=7)
    assert 7.0 < small.stderr / large.stderr < 14.0


def test_monte_carlo_is_reproducible_with_seed():
    a = mc_price(**PARAMS, n_paths=50_000, seed=99)
    b = mc_price(**PARAMS, n_paths=50_000, seed=99)
    assert a.price == b.price and a.stderr == b.stderr


def test_all_three_methods_agree():
    """The headline result of the notebook, asserted as a test."""
    exact = float(bs_price(**PARAMS))
    tree = binomial_price(**PARAMS, steps=3000)
    mc = mc_price(**PARAMS, n_paths=1_000_000, seed=2024)
    assert tree == pytest.approx(exact, abs=2e-3)
    assert mc.price == pytest.approx(exact, abs=4 * mc.stderr)


def test_mc_antithetic_requires_at_least_two_paths():
    with pytest.raises(ValueError):
        mc_price(**PARAMS, n_paths=1, antithetic=True)
