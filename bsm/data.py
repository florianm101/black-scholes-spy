"""Market data loading with a deterministic offline fallback.

Every loader here attempts live yfinance data first and falls back to
synthetic-but-realistic data if the network is unavailable, so the notebook and
test suite always run end to end. The `live` flag on each return value records
which path was taken.

Caveats on the live path, which are worth stating:
yfinance option quotes are delayed and not synchronised with the spot price;
the risk-free rate here is a single 13-week bill yield rather than a
term-matched curve; the dividend yield is a trailing-twelve-month
approximation rather than the market-implied forward yield.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .implied_vol import implied_vol

__all__ = ["MarketSnapshot", "load_underlying", "build_iv_surface"]

TRADING_DAYS = 252


@dataclass
class MarketSnapshot:
    """Everything the model needs about the underlying at one point in time."""

    ticker: str
    spot: float
    realized_vol: float
    risk_free: float
    div_yield: float
    history: pd.DataFrame
    live: bool
    log_returns: pd.Series = field(repr=False, default=None)

    def __post_init__(self):
        if self.log_returns is None:
            self.log_returns = (
                np.log(self.history["Close"] / self.history["Close"].shift(1)).dropna()
            )

    def summary(self) -> str:
        src = "live (yfinance)" if self.live else "SYNTHETIC FALLBACK"
        return (
            f"{self.ticker}  [{src}]\n"
            f"  spot           {self.spot:,.2f}\n"
            f"  realised vol   {self.realized_vol:.2%}  (annualised, daily log returns)\n"
            f"  risk-free r    {self.risk_free:.2%}\n"
            f"  div yield q    {self.div_yield:.2%}\n"
            f"  observations   {len(self.history)}"
        )


def _synthetic_history(n_days=504, s0=450.0, mu=0.10, sigma=0.16, seed=42):
    """Generate a GBM path as an offline stand-in for real price history.

    NOTE: returns are normal BY CONSTRUCTION, so any fat-tail analysis run on
    this data will correctly find nothing. That is a property of the fallback,
    not a finding about markets.
    """
    rng = np.random.default_rng(seed)
    mu_d, sig_d = mu / TRADING_DAYS, sigma / np.sqrt(TRADING_DAYS)
    shocks = rng.standard_normal(n_days)
    path = s0 * np.exp(np.cumsum(mu_d - 0.5 * sig_d**2 + sig_d * shocks))
    idx = pd.bdate_range(end=dt.date.today(), periods=n_days)
    return pd.DataFrame({"Close": path}, index=idx)


def load_underlying(ticker="SPY", period="2y", fallback_rate=0.04, fallback_q=0.012):
    """Load spot, history, realised vol, risk-free rate and dividend yield.

    Returns
    -------
    MarketSnapshot
    """
    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, auto_adjust=True)
        if hist.empty:
            raise RuntimeError("empty price history returned")

        # Yahoo emits a placeholder bar for the current session before any
        # trade data exists.
        hist = hist.dropna(subset=["Close"])
        if len(hist) < 50:
            raise RuntimeError(f"only {len(hist)} usable rows after cleaning")

        close = hist["Close"]
        spot = float(close.iloc[-1])
        log_ret = np.log(close / close.shift(1)).dropna()
        rvol = float(log_ret.std() * np.sqrt(TRADING_DAYS))

        try:  # ^IRX quotes the 13-week bill as an annualised percentage
            irx = yf.Ticker("^IRX").history(period="5d")
            rate = float(irx["Close"].iloc[-1]) / 100
        except Exception:
            rate = fallback_rate

        try:
            divs = tk.dividends
            if divs.empty:
                raise RuntimeError("no dividend history")
            trailing = divs[divs.index > divs.index[-1] - pd.Timedelta(days=365)]
            qy = float(trailing.sum() / spot)
            if not np.isfinite(qy) or qy <= 0.0:
                raise RuntimeError(f"invalid dividend yield {qy}")
        except Exception:
            qy = fallback_q

        return MarketSnapshot(ticker, spot, rvol, rate, qy, hist, live=True)

    except Exception as exc:
        print(f"[offline] yfinance unavailable ({type(exc).__name__}); using synthetic data.")
        hist = _synthetic_history()
        spot = float(hist["Close"].iloc[-1])
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        rvol = float(log_ret.std() * np.sqrt(TRADING_DAYS))
        return MarketSnapshot(
            ticker, spot, rvol, fallback_rate, fallback_q, hist, live=False
        )


def _synthetic_surface(seed=42):
    """A skewed smile with realistic index characteristics, for offline use."""
    rng = np.random.default_rng(seed)
    rows = []
    for T in [0.05, 0.12, 0.25, 0.5, 0.75, 1.0]:
        for m in np.linspace(0.78, 1.22, 28):
            k = np.log(m)
            atm = 0.14 + 0.03 * np.sqrt(T)      # upward ATM term structure
            skew = -0.28 / (T**0.35)            # skew steepest short-dated
            curve = 0.55 / (T**0.25)            # smile curvature
            iv = atm + skew * k + curve * k**2 + rng.normal(0, 0.003)
            rows.append((T, m, max(iv, 0.05)))
    return pd.DataFrame(rows, columns=["T", "moneyness", "iv"])


def build_iv_surface(
    snapshot: MarketSnapshot,
    max_expiries=6,
    min_days=10,
    max_years=1.5,
    moneyness_range=(0.75, 1.25),
    max_rel_spread=0.40,
    min_bid=0.05,
    min_points=30,
):
    """Build an implied volatility surface from live option chains.

    Data hygiene matters more than the maths here. Filters applied:
      * out-of-the-money options only (puts below spot, calls above), their
        premium is pure time value, so it carries maximum vol information
      * a live bid above `min_bid`
      * relative bid-ask spread below `max_rel_spread`
      * expiries between `min_days` and `max_years` (avoids microstructure
        noise and pin risk at the very front)
      * mid price, not the stale `lastPrice` field

    Returns
    -------
    (pd.DataFrame, bool)
        DataFrame with columns [T, moneyness, iv, kind], and a `live` flag.
    """
    if snapshot.live:
        try:
            import yfinance as yf

            tk = yf.Ticker(snapshot.ticker)
            today = pd.Timestamp.today()
            candidates = []
            for e in tk.options:
                T = (pd.Timestamp(e) - today).days / 365.0
                if min_days / 365 <= T <= max_years:
                    candidates.append((e, T))
            if len(candidates) > max_expiries:
                idxs = np.linspace(0, len(candidates) - 1, max_expiries).astype(int)
                candidates = [candidates[i] for i in idxs]

            rows = []
            for expiry, T in candidates:
                chain = tk.option_chain(expiry)
                for df, kind in [(chain.puts, "put"), (chain.calls, "call")]:
                    df = df.copy()
                    df["mid"] = 0.5 * (df["bid"] + df["ask"])
                    mny = df["strike"] / snapshot.spot
                    otm = (mny < 1.0) if kind == "put" else (mny > 1.0)
                    keep = (
                        otm
                        & (df["bid"] > min_bid)
                        & (df["ask"] > df["bid"])
                        & ((df["ask"] - df["bid"]) / df["mid"] < max_rel_spread)
                        & mny.between(*moneyness_range)
                    )
                    for _, row in df[keep].iterrows():
                        iv = implied_vol(
                            float(row["mid"]),
                            snapshot.spot,
                            float(row["strike"]),
                            T,
                            snapshot.risk_free,
                            snapshot.div_yield,
                            kind,
                        )
                        if not np.isnan(iv) and 0.03 < iv < 1.5:
                            rows.append((T, float(row["strike"]) / snapshot.spot, iv, kind))

            if len(rows) < min_points:
                raise RuntimeError(f"only {len(rows)} usable quotes after filtering")

            surf = pd.DataFrame(rows, columns=["T", "moneyness", "iv", "kind"])
            return surf.sort_values(["T", "moneyness"]).reset_index(drop=True), True

        except Exception as exc:
            print(f"[offline] option chain unavailable ({type(exc).__name__}: {exc})")

    surf = _synthetic_surface()
    surf["kind"] = np.where(surf["moneyness"] < 1.0, "put", "call")
    return surf.sort_values(["T", "moneyness"]).reset_index(drop=True), False
