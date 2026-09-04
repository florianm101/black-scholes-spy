"""Tests for the implied volatility solver."""

import numpy as np
import pytest

from bsm import bs_greeks, bs_price, implied_vol, no_arbitrage_bounds


@pytest.mark.parametrize("true_sigma", [0.08, 0.15, 0.30, 0.65, 1.20])
@pytest.mark.parametrize("K", [85, 100, 115])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_round_trip_recovers_input_vol(true_sigma, K, kind):
    """Price at a known vol, then invert. Skips vega-degenerate cases where
    the price carries no volatility information at all."""
    S, T, r, q = 100.0, 0.75, 0.04, 0.01
    vega = float(bs_greeks(S, K, T, r, true_sigma, q, kind)["vega"])
    price = float(bs_price(S, K, T, r, true_sigma, q, kind))
    iv = implied_vol(price, S, K, T, r, q, kind)

    if vega < 1e-6:
        pytest.skip("vega-degenerate: implied vol is not identifiable")
    assert iv == pytest.approx(true_sigma, abs=1e-6)


def test_reprices_to_input_even_when_vol_not_identifiable():
    """Weaker but always-true invariant: whatever the solver returns must
    reprice the option to the input price."""
    S, K, T, r, q = 100.0, 100.0, 0.5, 0.04, 0.0
    for sigma in np.linspace(0.05, 1.5, 20):
        price = float(bs_price(S, K, T, r, sigma, q, "call"))
        iv = implied_vol(price, S, K, T, r, q, "call")
        assert not np.isnan(iv)
        assert float(bs_price(S, K, T, r, iv, q, "call")) == pytest.approx(price, abs=1e-7)


def test_price_below_intrinsic_returns_nan():
    """A price violating the no-arbitrage lower bound has no implied vol."""
    S, K, T, r = 100.0, 80.0, 1.0, 0.05
    lower, _ = no_arbitrage_bounds(S, K, T, r, 0.0, "call")
    assert np.isnan(implied_vol(lower - 1.0, S, K, T, r, 0.0, "call"))


def test_price_above_upper_bound_returns_nan():
    """A call cannot be worth more than the (dividend-discounted) spot."""
    assert np.isnan(implied_vol(200.0, 100.0, 90.0, 1.0, 0.05, 0.0, "call"))


def test_newton_converges_quickly_for_typical_atm_option():
    """Newton should need very few iterations near the money -- that is the
    whole point of using the analytical vega as the derivative."""
    S, K, T, r = 100.0, 100.0, 0.25, 0.04
    price = float(bs_price(S, K, T, r, 0.22))
    iv_full = implied_vol(price, S, K, T, r, max_iter=100)
    iv_few = implied_vol(price, S, K, T, r, max_iter=6)
    assert iv_few == pytest.approx(iv_full, abs=1e-8)


def test_bisection_fallback_reached_when_newton_blocked():
    """Force Newton to fail via an absurd starting point; bisection must
    still bracket and find the root."""
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    price = float(bs_price(S, K, T, r, 0.25))
    iv = implied_vol(price, S, K, T, r, sigma_init=1e-9, tol=1e-10)
    assert iv == pytest.approx(0.25, abs=1e-5)


def test_deep_itm_short_dated_is_documented_degenerate_case():
    """Deep ITM short-dated: price == intrinsic to machine precision, vega ~ 0.
    The solver must not crash; the recovered number is meaningless and the
    test asserts only that vega really is degenerate."""
    S, K, T, r = 100.0, 50.0, 0.05, 0.05
    price = float(bs_price(S, K, T, r, 0.20))
    vega = float(bs_greeks(S, K, T, r, 0.20)["vega"])
    iv = implied_vol(price, S, K, T, r)
    assert vega < 1e-10
    assert np.isnan(iv)

def test_identifiable_options_are_not_rejected():
    """The degeneracy gate must not fire on options with real vega."""
    S, T, r = 100.0, 0.25, 0.05
    for K in [80, 90, 100, 110, 120]:
        price = float(bs_price(S, K, T, r, 0.20))
        assert implied_vol(price, S, K, T, r) == pytest.approx(0.20, abs=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_no_arbitrage_bounds_bracket_the_price(kind):
    S, K, T, r, q = 100.0, 105.0, 0.8, 0.045, 0.02
    lo, hi = no_arbitrage_bounds(S, K, T, r, q, kind)
    for sigma in [0.05, 0.3, 2.0]:
        price = float(bs_price(S, K, T, r, sigma, q, kind))
        assert lo - 1e-9 <= price <= hi + 1e-9
