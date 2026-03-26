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

- `Euro500_Portfolio.ipynb`: Builds and analyzes the Euro500 equity portfolio at the firm level.
- `LSEG_DataPull_DailyReturns.ipynb`: Pulls daily price/return data for the Euro500 firm universe (including firm_id handling).
- `Euro500_IndexReturns.ipynb`: Builds and analyzes Euro500 index returns as the market/benchmark return series.
- `LSEG_DataPull_Macaulay.ipynb`: Pulls the LSEG raw inputs required for Macaulay-based equity duration.
- `LSEG_DataPull_Netpayout.ipynb`: Pulls the LSEG raw inputs required for net-payout-based duration.
- `EQDuration_Macaulay.ipynb`: Computes equity duration using a Macaulay-style approach based on prepared firm-level data.
- `EQDuration_NetPayout.ipynb`: Computes equity duration using a net-payout approach.
- `ECBShocks_Equitys_Regressions.ipynb`: Runs firm-level panel regressions of ECB shocks on equity returns with duration interactions.
- `ECBShocks_Index_Regressions.ipynb`: Runs index-level regressions of ECB shocks on Euro500 and benchmark index returns.

## Python Files

- `lseg_series_puller.py`: Shared pull engine for LSEG series and daily returns (batching, caching, resume/checkpoint, bad-id handling).
- `plot_style.py`: Shared plotting style helpers used across notebooks.

