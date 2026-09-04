"""Black-Scholes-Merton option pricing, Greeks, implied volatility and surfaces.

Quick start
-----------
>>> from bsm import bs_price, bs_greeks, implied_vol
>>> bs_price(100, 100, 1.0, 0.05, 0.20)
10.450583...
"""

from .analytical import bs_greeks, bs_price, d1_d2
from .data import MarketSnapshot, build_iv_surface, load_underlying
from .implied_vol import implied_vol, no_arbitrage_bounds
from .numerical import MCResult, binomial_price, mc_price

__version__ = "1.0.0"

__all__ = [
    "bs_price",
    "bs_greeks",
    "d1_d2",
    "implied_vol",
    "no_arbitrage_bounds",
    "binomial_price",
    "mc_price",
    "MCResult",
    "load_underlying",
    "build_iv_surface",
    "MarketSnapshot",
]
