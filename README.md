# Black–Scholes on SPY

[![tests](https://github.com/florianm101/black-scholes-spy/actions/workflows/tests.yml/badge.svg)](https://github.com/florianm101/black-scholes-spy/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![notebook](https://img.shields.io/badge/notebook-view-orange)](https://github.com/florianm101/black-scholes-spy/blob/main/notebooks/black_scholes_spy.ipynb)

An implementation and empirical stress-test of the Black–Scholes–Merton model on real S&P 500 (SPY) data: closed-form pricing, analytical Greeks, implied volatility inversion, three independent pricing methods shown to converge, and a fitted implied volatility surface that demonstrates exactly where the model's assumptions fail.

The point of the project is not that Black–Scholes prices options. It's that **it doesn't**, and that the shape of its failure, visible in the volatility surface, is more informative than the model itself.

---

## What's here

| Component | Location | Summary |
|---|---|---|
| Pricing & Greeks | `bsm/analytical.py` | Closed-form BSM with continuous dividend yield; all five Greeks derived analytically |
| Implied volatility | `bsm/implied_vol.py` | Newton–Raphson with bisection fallback, no-arbitrage bounds, documented degenerate cases |
| Numerical methods | `bsm/numerical.py` | CRR binomial tree (European + American), Monte Carlo with antithetic variates |
| Market data | `bsm/data.py` | yfinance loader with option-chain filtering and a deterministic offline fallback |
| Narrative analysis | `notebooks/black_scholes_spy.ipynb` | The full write-up: theory, interactive dials, convergence study, vol surface, fat-tail analysis |
| Test suite | `tests/` | 112 tests covering pricing identities, Greek accuracy, solver behaviour and convergence |

---

## Quick start

```bash
git clone https://github.com/florianm101/black-scholes-spy.git
cd black-scholes-spy
pip install -e ".[dev,data,notebook]"
```

```bash
pytest                                          # run the test suite
jupyter lab notebooks/black_scholes_spy.ipynb
```

```python
from bsm import bs_price, bs_greeks, implied_vol, mc_price

bs_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
# 10.450583572185565   (matches the standard textbook benchmark)

bs_greeks(100, 100, 1.0, 0.05, 0.20)["delta"]
# 0.6368306511756191

implied_vol(price=10.4506, S=100, K=100, T=1.0, r=0.05)
# 0.19999984...

mc = mc_price(100, 100, 1.0, 0.05, 0.20, n_paths=1_000_000, seed=42)
mc.price, mc.ci95
```

`theta` is returned per year (divide by 365 for the per-day figure desks quote) and `vega` per unit of vol (divide by 100 for per-vol-point).

---

## Correctness

Analytical results are worth nothing unless they're checked, so the model is validated four independent ways:

1. **Textbook benchmark.** $S=K=100$, $T=1$, $r=5\%$, $\sigma=20\%$ reproduces the standard call/put values (10.4506 / 5.5735).
2. **Put–call parity.** $C - P = Se^{-qT} - Ke^{-rT}$ holds to $10^{-10}$ across strikes, maturities and dividend yields.
3. **Greeks vs. finite differences.** Every analytical Greek agrees with a central difference to $\sim10^{-5}$ across a parameter grid. This is the test that catches errors in the hand-derived calculus.
4. **The Black–Scholes PDE itself.** The residual $\theta + \tfrac{1}{2}\sigma^2S^2\Gamma + (r-q)S\Delta - rV$ is zero to $10^{-8}$, confirming prices and Greeks are mutually consistent.

Plus convergence of the numerical methods to the closed form, and the classic early-exercise results (an American call on a non-dividend payer equals its European counterpart; an American put strictly exceeds it).

**112 passed**


---

## Selected results

**Three methods, one price.** A 3-month ATM call priced independently:

| Method | Price | Note |
|---|---|---|
| Closed-form BSM | reference | exact |
| CRR binomial, 3000 steps | agrees to <2e-3 | error $O(1/n)$, with the classic odd/even oscillation |
| Monte Carlo, 1M paths | within its 95% CI | error $O(1/\sqrt{N})$ |

Antithetic variates cut the Monte Carlo standard error by roughly **1.4×** at equal path count. That is asserted as a test, not just claimed.

**The volatility surface.** Implied vols fitted across strikes and maturities from filtered SPY option chains show the three features Black–Scholes cannot produce: a strongly negative skew (steepest at short maturities), positive convexity in log-moneyness at every expiry, and a non-flat ATM term structure.

One precision worth flagging, since the folk description gets it wrong. For an **equity index** the skew term dominates, so implied vol is close to monotonically *decreasing* in strike over the liquid range: a smirk, not a symmetric smile. Both wings sitting above ATM is a currency-market phenomenon. What survives for indices is the convexity.

**Fat tails.** SPY's own daily log returns are tested against normality (excess kurtosis, Jarque–Bera, tail-exceedance counts vs. theoretical frequencies, Q–Q plot). Under normality a daily move beyond 4σ should appear roughly once every ~63 years of trading. It doesn't work out that way.

---

## Design notes

**The implied vol solver returns `nan` on purpose.** Deep-ITM short-dated options price to discounted intrinsic at machine precision. Vega is numerically zero, so the price carries *no information about volatility* and no solver can recover it; every candidate σ reprices the option within tolerance. Rather than return a confident meaningless number, the solver reports failure and the notebook demonstrates the case explicitly. This is also why `build_iv_surface` uses OTM quotes only: an OTM premium is pure time value, hence pure volatility information.

**The offline fallback is flagged, never disguised.** `load_underlying` and `build_iv_surface` return a `live` boolean, and `MarketSnapshot.summary()` prints `SYNTHETIC FALLBACK` when the network is unavailable. The synthetic price path is GBM, so its returns are normal *by construction*, and the fat-tail section correctly finds nothing and says so. Presenting simulated data as market evidence would be the worst failure mode this project could have.

**Known limitations of the live data path**, stated rather than buried: yfinance option quotes are delayed and unsynchronised with spot; mid-prices from wide markets are noisy; the risk-free rate is a single 13-week bill yield rather than a term-matched curve; the dividend yield is a trailing-twelve-month approximation rather than the market-implied forward yield.

**`load_underlying` validates what it fetches rather than trusting it.** Yahoo emits a placeholder bar for the current session before any trades print, which would otherwise make spot `NaN` and poison every downstream price. Incomplete rows are dropped, and any derived field that comes back implausible routes to the synthetic fallback rather than returning a snapshot labelled `live=True` with bad numbers inside it. The mislabelling is the worse failure.

---

## Extensions

Roughly ascending in difficulty:

- **American options.** Supported via `binomial_price(..., american=True)`; quantify the early-exercise premium across moneyness and dividend yield.
- **Local volatility** (Dupire, 1994). A $\sigma(S,t)$ that reprices the whole surface exactly.
- **Stochastic volatility** (Heston, 1993). Mean-reverting variance correlated with spot; generates skew endogenously. Calibrating it to the surface built here is the natural next step.
- **Jump diffusion** (Merton, 1976). Poisson jumps on GBM, aimed at the fat-tail failure directly.
- **SVI parameterisation** (Gatheral). An arbitrage-free functional fit per expiry, replacing raw scatter.

---

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy 81(3).
- Merton, R. C. (1973). *Theory of Rational Option Pricing.* Bell Journal of Economics 4(1).
- Merton, R. C. (1976). *Option Pricing When Underlying Stock Returns Are Discontinuous.* Journal of Financial Economics 3.
- Cox, J., Ross, S. & Rubinstein, M. (1979). *Option Pricing: A Simplified Approach.* Journal of Financial Economics 7.
- Heston, S. (1993). *A Closed-Form Solution for Options with Stochastic Volatility.* Review of Financial Studies 6(2).
- Dupire, B. (1994). *Pricing with a Smile.* Risk 7(1).
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide.* Wiley.

## License

MIT. See [LICENSE](LICENSE).