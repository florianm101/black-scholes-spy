# Rendered output

`black_scholes_spy.html` is a static export of the notebook with all figures and
printed output embedded, readable in any browser without a Python environment.

Two caveats:

1. **The sliders don't work in HTML.** `ipywidgets` needs a live kernel. Run the
   notebook itself for the interactive volatility and rate dials.
2. **Check the data source line.** The notebook prints either `live (yfinance)`
   or `SYNTHETIC FALLBACK` near the top. If this export was generated without
   network access, the numbers come from a simulated GBM path. Realistic in
   shape, but not market data, and the fat-tail section will correctly find
   nothing because GBM returns are normal by construction.

Regenerate with live data:

```bash
make render
```
