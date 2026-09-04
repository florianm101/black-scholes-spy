"""Tests for the data layer.

These must pass with or without network access, which is the entire point of
the fallback design.
"""

import numpy as np

from bsm import build_iv_surface, load_underlying
from bsm.data import _synthetic_history, _synthetic_surface


def test_synthetic_history_is_reproducible_and_positive():
    a = _synthetic_history(seed=1)
    b = _synthetic_history(seed=1)
    assert np.allclose(a["Close"], b["Close"])
    assert (a["Close"] > 0).all()
    assert len(a) == 504


def test_load_underlying_returns_usable_snapshot_either_way():
    snap = load_underlying("SPY", period="1y")
    assert snap.spot > 0
    assert 0.01 < snap.realized_vol < 2.0
    assert 0.0 <= snap.risk_free < 0.25
    assert 0.0 <= snap.div_yield < 0.15
    assert len(snap.history) > 100
    assert isinstance(snap.live, bool)
    assert "spot" in snap.summary()


def test_snapshot_flags_synthetic_data_in_summary():
    snap = load_underlying("SPY")
    if not snap.live:
        assert "SYNTHETIC" in snap.summary()


def test_log_returns_derived_correctly():
    snap = load_underlying("SPY", period="1y")
    assert len(snap.log_returns) == len(snap.history) - 1
    assert np.isfinite(snap.log_returns).all()


def test_synthetic_surface_exhibits_negative_skew():
    """OTM puts must price at higher implied vol than OTM calls -- the
    defining feature of an equity index surface."""
    surf = _synthetic_surface()
    for T, grp in surf.groupby("T"):
        put_wing = grp[grp["moneyness"] < 0.90]["iv"].mean()
        call_wing = grp[grp["moneyness"] > 1.10]["iv"].mean()
        assert put_wing > call_wing, f"no skew at T={T}"


def test_synthetic_surface_is_convex_in_log_moneyness():
    """Convexity (positive quadratic term) is the correct test of a 'smile'.

    Note it is NOT true for an equity index that both wings sit above ATM: the
    skew term dominates, so index vol is close to monotonically decreasing in
    strike over the liquid range. Convexity
    is what survives.
    """
    surf = _synthetic_surface()
    for T, grp in surf.groupby("T"):
        quad = np.polyfit(np.log(grp["moneyness"]), grp["iv"], 2)[0]
        assert quad > 0, f"not convex at T={T}"


def test_synthetic_surface_atm_term_structure_is_upward_sloping():
    surf = _synthetic_surface()
    atm = [
        np.interp(1.0, g.sort_values("moneyness")["moneyness"], g.sort_values("moneyness")["iv"])
        for _, g in surf.groupby("T")
    ]
    assert atm[-1] > atm[0]


def test_build_iv_surface_returns_valid_frame():
    snap = load_underlying("SPY")
    surf, live = build_iv_surface(snap)
    assert set(["T", "moneyness", "iv"]).issubset(surf.columns)
    assert len(surf) >= 30
    assert (surf["iv"] > 0).all() and (surf["iv"] < 2.0).all()
    assert (surf["T"] > 0).all()
    assert isinstance(live, bool)

def test_incomplete_trailing_bar_is_dropped():
    """Yahoo returns a NaN placeholder bar for the current session. It must
    not propagate into spot, which would poison every downstream price."""
    snap = load_underlying("SPY", period="1y")
    assert np.isfinite(snap.spot)
    assert np.isfinite(snap.div_yield)
    assert snap.history["Close"].notna().all()
