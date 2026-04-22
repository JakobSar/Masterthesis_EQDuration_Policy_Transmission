# Equity Duration and the Monetary Policy Transmission

This repository contains the code, data pipeline, and empirical analysis for my MSc Economics master's thesis. The project estimates firm-level equity duration for a European large-cap universe ("Euro500") and studies how duration shapes the transmission of ECB monetary policy shocks to equity returns.

## Overview

The pipeline is organized into four stages:

1. **Universe & portfolio construction** — build the Euro500 firm universe from STOXX Europe 600 membership, and compute daily returns and an index benchmark.
2. **LSEG data pulls** — fetch fundamentals, analyst forecasts, and daily prices from Refinitiv/LSEG `lseg-data` with batching, caching, and resume/checkpoint support.
3. **Equity duration estimation** — compute duration measures using (a) a net-payout approach, (b) an analyst-forecast-based clean-surplus framework with Jensen-inequality corrections, and (c) simpler proxies for robustness (shareholder yield, B/M, E/P).
4. **Regressions** — estimate firm-level panel regressions and index-level regressions of equity returns on high-frequency ECB monetary policy shocks, interacted with equity duration.

## Repository structure

```
Project_Code/              # this repository (notebooks + shared Python modules)
Project_Data/              # data root (not versioned; see "Data layout")
  ├── intermediate/        # parquet outputs from the pipeline
  ├── cache/               # per-firm LSEG pull caches
  ├── graphs/              # figures by notebook
  └── tables/              # regression tables by notebook
```

`Project_Data/` sits next to `Project_Code/` by default. Override with the `PROJECT_DATA_DIR` environment variable — see [project_paths.py](project_paths.py).

## Notebooks

Run notebooks in the order below; each stage consumes the outputs of the previous stage.

### 1. Data & portfolio construction
- [Euro500_Portfolio.ipynb](Euro500_Portfolio.ipynb) — builds and analyzes the Euro500 equity portfolio at the firm level.
- [LSEG_DataPull_DailyReturns.ipynb](LSEG_DataPull_DailyReturns.ipynb) — pulls daily price/return data for the Euro500 firm universe (including firm_id handling).
- [Euro500_IndexReturns.ipynb](Euro500_IndexReturns.ipynb) — builds and analyzes Euro500 index returns as the market/benchmark return series.

### 2. LSEG data pulls
- [LSEG_DataPull_Netpayout.ipynb](LSEG_DataPull_Netpayout.ipynb) — pulls the LSEG raw inputs (balance-sheet, income-statement, cashflow/payout items) required for net-payout-based duration.
- [LSEG_DataPull_AnalystBased.ipynb](LSEG_DataPull_AnalystBased.ipynb) — pulls analyst consensus forecast data (EPS FY1–FY5, long-term growth, DPS) for the Euro500 firm universe.

### 3. Equity duration
- [EQDuration_NetPayout.ipynb](EQDuration_NetPayout.ipynb) — computes equity duration using a net-payout approach.
- [EQDuration_AnalystBased.ipynb](EQDuration_AnalystBased.ipynb) — computes equity duration from analyst consensus forecasts using the clean-surplus framework with Jensen-inequality corrections.
- [EQDuration_Robustness.ipynb](EQDuration_Robustness.ipynb) — computes alternative equity duration proxies using shareholder yield, book-to-market ratio, and earnings-to-price ratio.

### 4. Regressions
- [ECBShocks_Equities_Regressions.ipynb](ECBShocks_Equities_Regressions.ipynb) — firm-level panel regressions of ECB monetary policy shocks on equity returns with duration interactions.
- [ECBShocks_Index_Regressions.ipynb](ECBShocks_Index_Regressions.ipynb) — index-level regressions of ECB shocks on Euro500 and benchmark index returns.

## Shared Python modules

- [project_paths.py](project_paths.py) — central project directory paths for data, cache, graphs, and tables. Honors the `PROJECT_DATA_DIR` environment variable.
- [lseg_series_puller.py](lseg_series_puller.py) — shared pull engine for LSEG series and daily returns: batching, per-firm parquet caching, candidate resolution (ISIN → RIC_current → RIC → pull_id history), resume/checkpointing, and bad-id handling with cooldown.
- [plot_style.py](plot_style.py) — shared matplotlib plotting style helpers used across notebooks.

## Environment

Python 3.13 with a dedicated virtual environment.

```bash
python3.13 -m venv .venv_thesis
ln -s .venv_thesis .venv        # optional: lets tools auto-detect .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verification:

```bash
python -c "import pandas, numpy, statsmodels, lseg.data; print(pandas.__version__)"
```

Notes:
- Prefer `python -m pip ...` to avoid mixing `pip` from a different interpreter.
- The LSEG data layer requires a valid `lseg-data` configuration file / API credentials in your user profile. Credentials are **not** part of this repository.

Optional auto-activation with direnv:

```bash
brew install direnv
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
direnv allow .
```

## Data layout

Intermediate parquet files written to `Project_Data/intermediate/` (selection):

| File | Produced by | Content |
|---|---|---|
| `stoxx600_membership_matrix_1999_2025_eurohq.parquet` | Portfolio construction | STOXX 600 membership matrix, Euro-HQ firms |
| `euro500.parquet` | `Euro500_Portfolio.ipynb` | Euro500 firm universe |
| `euro500_daily_returns.parquet` | `LSEG_DataPull_DailyReturns.ipynb` | Firm-level daily returns |
| `euro500_index_returns.parquet` | `Euro500_IndexReturns.ipynb` | Euro500 index benchmark returns |
| `euro500_netpayout.parquet` | `LSEG_DataPull_Netpayout.ipynb` | Balance-sheet / income / payout inputs |
| `euro500_analystbased.parquet` | `LSEG_DataPull_AnalystBased.ipynb` | Analyst consensus inputs |
| `EQDuration_NetPayout.parquet` | `EQDuration_NetPayout.ipynb` | Net-payout-based duration |
| `EQDuration_Fcst.parquet` | `EQDuration_AnalystBased.ipynb` | Analyst-forecast-based duration |
| `EQDuration_Robustness.parquet` | `EQDuration_Robustness.ipynb` | Robustness proxies |
| `shocks_ecb_mpd_me_d.csv` | External input | High-frequency ECB monetary policy shocks |
| `rates_2yOIS_daily.parquet` | External input | Euro-area 2y OIS rates |

The `cache/` subdirectory holds per-firm parquet caches used by the LSEG pullers to avoid re-fetching unchanged (firm, date) combinations.

## Reproducing the analysis

From a clean state (valid LSEG credentials and input shock/rate files in place):

1. `Euro500_Portfolio.ipynb`
2. `LSEG_DataPull_DailyReturns.ipynb` → `Euro500_IndexReturns.ipynb`
3. `LSEG_DataPull_Netpayout.ipynb` and `LSEG_DataPull_AnalystBased.ipynb`
4. `EQDuration_NetPayout.ipynb`, `EQDuration_AnalystBased.ipynb`, `EQDuration_Robustness.ipynb`
5. `ECBShocks_Equities_Regressions.ipynb`, `ECBShocks_Index_Regressions.ipynb`

LSEG rate limiting: the data layer enforces a shared quota across steps. If a pull returns HTTP 429, wait 5–10 minutes and resume — each puller checkpoints and skips already-cached (firm, date) combinations.

## Data availability

Raw LSEG / Refinitiv data is proprietary and cannot be redistributed. The repository therefore contains only code; the `Project_Data/` tree is recreated locally via the pull notebooks against your own LSEG entitlement. ECB high-frequency shock data is taken from the ECB's Euro-Area Monetary Policy Event-Study Database (EA-MPD).

## Citation

If you use this code, please cite the thesis:

> Sarrazin, J. *Equity Duration and the Monetary Policy Transmission.* MSc Economics master's thesis.

## License

Code in this repository is released under the MIT License (see `LICENSE` once added). Underlying market data from LSEG/Refinitiv and the ECB EA-MPD are subject to their respective providers' terms.
