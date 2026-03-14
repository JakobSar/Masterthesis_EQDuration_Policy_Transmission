# Equity Duration and the Foreign Policy Transmission

This repository contains the code and analysis for my MSc Economics master thesis.

## Notebooks

- `LSEG_DataPull_DailyReturns.ipynb`: Pullt tägliche Kursdaten/Returns für das Euro500-Firmenuniversum (inkl. firm_id-Logik).
- `LSEG_DataPull_Macaulay.ipynb`: Lädt die Rohdaten aus LSEG, die für die Macaulay-basierte Equity-Duration benötigt werden.
- `LSEG_DataPull_Netpayout.ipynb`: Lädt die Rohdaten aus LSEG für die Net-Payout-basierte Duration.
- `EQDuration_Macaulay.ipynb`: Berechnet Equity Duration mit einem Macaulay-Ansatz auf Basis der aufbereiteten Firmendaten.
- `EQDuration_NetPayout.ipynb`: Berechnet Equity Duration mit einem Net-Payout-Ansatz.
- `Euro500_IndexReturns.ipynb`: Erstellt/analysiert Index-Returns für den Euro500 als Markt-/Benchmark-Reihe.
- `Euro500_Portfolio.ipynb`: Baut und analysiert das Euro500-Aktienportfolio auf Firmenebene.
- `ECBShocks_Regressions.ipynb`: Führt Panel-Regressionen von ECB-Schocks auf Renditen mit Duration-Interaktion durch.

