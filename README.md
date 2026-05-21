# mse342-project

## Daily Return Data

Use the Kenneth French 49 industry portfolio daily returns as the no-wait data
source. The series are CRSP-backed, long-history, and already returned as daily
portfolio returns, which avoids survivorship and ticker-cleaning work.

Build the local dataset:

```bash
../venv/bin/python scripts/build_french_returns.py
```

This writes:

- `data/processed/french49_daily_returns.csv`
- `data/processed/french49_daily_returns.metadata.json`
- `CSDI/data/french49_daily/data.pkl`
- `CSDI/data/french49_daily/meanstd.pkl`
- `CSDI/data/french49_daily/metadata.json`

The CSV uses decimal daily returns. By default it keeps value-weighted industry
returns from `1990-01-01` onward and drops the catch-all `Other` portfolio,
leaving 48 clean industry series.

Run the fixed-split CSDI scenario experiment:

```bash
bash CSDI_Experiment/scripts/run_fixed_split_cuda.sh
```

The experiment trains on data up to `TRAIN_END_DATE` and generates scenario
paths for the requested horizons. The build script needs NumPy. The CSDI run
also needs the packages in `CSDI/requirements.txt`, including PyTorch.
