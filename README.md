# Equity Duration and the Foreign Policy Transmission

This repository contains the code and analysis for my MSc Economics master thesis.

## Environment (important)

Use the project virtual environment for all Python commands.

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import pandas; print(pandas.__version__)"
```

Notes:
- `.venv` is intentionally linked to `.venv_thesis` so tools can auto-detect the environment.
- Prefer `python -m pip ...` to avoid mixing `pip` from a different interpreter.

Optional auto-activation with direnv:

```bash
brew install direnv
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
direnv allow .
```

## Notebooks

### Data & Portfolio Construction
- `Euro500_Portfolio.ipynb`: Builds and analyzes the Euro500 equity portfolio at the firm level.
- `LSEG_DataPull_DailyReturns.ipynb`: Pulls daily price/return data for the Euro500 firm universe (including firm_id handling).
- `Euro500_IndexReturns.ipynb`: Builds and analyzes Euro500 index returns as the market/benchmark return series.

### LSEG Data Pulls
- `LSEG_DataPull_Netpayout.ipynb`: Pulls the LSEG raw inputs required for net-payout-based duration.
- `LSEG_DataPull_Implied.ipynb`: Pulls and processes implied-duration variables including price, beta, market cap, and risk-free rates for the Euro500 universe.
- `LSEG_DataPull_AnalystBased.ipynb`: Pulls analyst consensus forecast data (EPS, DPS) for the Euro500 firm universe.

### Equity Duration
- `EQDuration_NetPayout.ipynb`: Computes equity duration using a net-payout approach.
- `EQDuration_Implied.ipynb`: Computes equity duration using a clean-surplus, accounting-based approach with finite-horizon cash flows and perpetuity.
- `EQDuration_AnalystBased.ipynb`: Computes equity duration based on analyst consensus forecasts using the clean-surplus framework with Jensen-inequality corrections.
- `EQDuration_Robustness.ipynb`: Computes alternative equity duration proxies using shareholder yield, book-to-market ratio, and earnings-to-price ratio.

### Regressions
- `ECBShocks_Equities_Regressions.ipynb`: Runs firm-level panel regressions of ECB shocks on equity returns with duration interactions.
- `ECBShocks_Index_Regressions.ipynb`: Runs index-level regressions of ECB shocks on Euro500 and benchmark index returns.

### Archived
- `alt/LSEG_DataPull_Macaulay.ipynb`: Pulls the LSEG raw inputs required for Macaulay-based equity duration (superseded).
- `alt/EQDuration_Macaulay.ipynb`: Computes equity duration using a Macaulay-style approach (superseded).
- `alt/STOXX_Constituents.ipynb`: STOXX constituent data exploration (superseded).

## Python Files

- `project_paths.py`: Defines central project directory paths for data, cache, graphs, and tables.
- `lseg_series_puller.py`: Shared pull engine for LSEG series and daily returns (batching, caching, resume/checkpoint, bad-id handling).
- `plot_style.py`: Shared plotting style helpers used across notebooks.

