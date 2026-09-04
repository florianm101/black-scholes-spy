"""Tests for closed-form pricing and analytical Greeks."""

import numpy as np
import pytest

from bsm import bs_greeks, bs_price

# Textbook benchmark: S=K=100, T=1, r=5%, sigma=20%, q=0
BENCH = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


def test_textbook_benchmark_call():
    assert bs_price(**BENCH, kind="call") == pytest.approx(10.450584, abs=1e-5)


def test_textbook_benchmark_put():
    assert bs_price(**BENCH, kind="put") == pytest.approx(5.573526, abs=1e-5)


@pytest.mark.parametrize("K", [70, 90, 100, 110, 140])
@pytest.mark.parametrize("T", [0.05, 0.5, 2.0])
@pytest.mark.parametrize("q", [0.0, 0.02])
def test_put_call_parity(K, T, q):
    """C - P = S*exp(-qT) - K*exp(-rT). Holds exactly, by construction."""
    S, r, sigma = 100.0, 0.05, 0.25
    c = bs_price(S, K, T, r, sigma, q, "call")
    p = bs_price(S, K, T, r, sigma, q, "put")
    expected = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert (c - p) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_price_within_no_arbitrage_bounds(kind):
    S, K, T, r, sigma, q = 100.0, 105.0, 0.75, 0.04, 0.3, 0.01
    price = float(bs_price(S, K, T, r, sigma, q, kind))
    fwd, disc_k = S * np.exp(-q * T), K * np.exp(-r * T)
    if kind == "call":
        assert max(fwd - disc_k, 0) <= price <= fwd
    else:
        assert max(disc_k - fwd, 0) <= price <= disc_k


def test_price_monotonic_in_vol():
    """Vega > 0 everywhere: both calls and puts gain value as vol rises."""
    vols = np.linspace(0.05, 1.5, 40)
    for kind in ("call", "put"):
        prices = bs_price(100, 100, 1.0, 0.05, vols, 0.0, kind)
        assert np.all(np.diff(prices) > 0)


def test_zero_vol_limit_is_discounted_intrinsic():
    """As sigma -> 0 the option converges to its forward intrinsic value."""
    S, K, T, r = 100.0, 90.0, 1.0, 0.05
    price = float(bs_price(S, K, T, r, 1e-8, 0.0, "call"))
    assert price == pytest.approx(S - K * np.exp(-r * T), abs=1e-6)


def test_vectorisation_over_strikes():
    strikes = np.linspace(80, 120, 25)
    out = bs_price(100, strikes, 1.0, 0.05, 0.2)
    assert out.shape == strikes.shape
    assert np.all(np.diff(out) < 0)  # call price decreasing in strike


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(
    "K,T,sigma", [(90, 0.25, 0.15), (100, 1.0, 0.20), (115, 2.0, 0.45)]
)
def test_greeks_match_finite_differences(kind, K, T, sigma):
    """Every analytical Greek must agree with a central finite difference.

    This is the single most important test in the suite: it catches any error
    in the hand-derived calculus.
    """
    S, r, q = 100.0, 0.05, 0.015
    g = bs_greeks(S, K, T, r, sigma, q, kind)

    def px(**kw):
        args = dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind=kind)
        args.update(kw)
        return float(bs_price(**args))

    # Steps scaled to each input's magnitude. Gamma needs a larger relative step
    # than the first-order Greeks: dividing a second difference by h**2 amplifies
    # floating-point cancellation, so the optimal step is ~eps**(1/4), not eps**(1/2).
    base = px()
    h_s, h_sig, h_T, h_r = S * 1e-4, 1e-5, 1e-5, 1e-6
    fd = {
        "delta": (px(S=S + h_s) - px(S=S - h_s)) / (2 * h_s),
        "gamma": (px(S=S + h_s) - 2 * base + px(S=S - h_s)) / h_s**2,
        "vega": (px(sigma=sigma + h_sig) - px(sigma=sigma - h_sig)) / (2 * h_sig),
        "theta": -(px(T=T + h_T) - px(T=T - h_T)) / (2 * h_T),
        "rho": (px(r=r + h_r) - px(r=r - h_r)) / (2 * h_r),
    }
    for name in fd:
        analytic = float(g[name])
        assert analytic == pytest.approx(fd[name], rel=1e-5), (
            f"{name}: analytic {analytic} vs finite difference {fd[name]}"
        )


def test_greek_sign_conventions():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
    c = bs_greeks(S, K, T, r, sigma, q, "call")
    p = bs_greeks(S, K, T, r, sigma, q, "put")
    assert 0 < c["delta"] < 1
    assert -1 < p["delta"] < 0
    assert c["gamma"] > 0 and p["gamma"] > 0
    assert c["vega"] > 0 and p["vega"] > 0
    assert c["theta"] < 0          # ATM long call decays
    assert c["rho"] > 0 > p["rho"]


def test_gamma_and_vega_identical_across_call_put():
    """Gamma and vega do not depend on the option type (put-call parity)."""
    a = bs_greeks(100, 105, 0.5, 0.04, 0.3, 0.01, "call")
    b = bs_greeks(100, 105, 0.5, 0.04, 0.3, 0.01, "put")
    assert a["gamma"] == pytest.approx(b["gamma"], rel=1e-12)
    assert a["vega"] == pytest.approx(b["vega"], rel=1e-12)


def test_deep_itm_call_delta_approaches_one():
    d = float(bs_greeks(100, 10, 1.0, 0.05, 0.2, 0.0, "call")["delta"])
    assert d == pytest.approx(1.0, abs=1e-6)


def test_bs_pde_is_satisfied():
    """theta + 0.5*sigma^2*S^2*gamma + (r-q)*S*delta - r*V = 0.

    The PDE the model is derived from. If prices and Greeks are mutually
    consistent, this residual is zero.
    """
    S, K, T, r, sigma, q = 100.0, 95.0, 0.6, 0.045, 0.28, 0.012
    for kind in ("call", "put"):
        V = float(bs_price(S, K, T, r, sigma, q, kind))
        g = bs_greeks(S, K, T, r, sigma, q, kind)
        residual = (
            float(g["theta"])
            + 0.5 * sigma**2 * S**2 * float(g["gamma"])
            + (r - q) * S * float(g["delta"])
            - r * V
        )
        assert residual == pytest.approx(0.0, abs=1e-8), f"{kind}: {residual}"


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0.05, 0.2, kind="straddle")
