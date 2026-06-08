# predictive-signal-auc-research

Offline leak-free backtests that measure whether candidate signals predict the winner of Polymarket BTC up/down 5-minute binary markets. Nothing here trades or touches the live bot.

Three studies: per-feature out-of-sample AUC, order-flow direction, and a mean-reversion fade window. All are no-leak: features come only from ticks at or before the decision point, the winner label is attached after feature extraction, and the fade study uses first-trigger semantics (no future-aware sorting or dedup).

## Scripts

- `sim_predictive_power.py`: per-feature univariate OOS AUC at the tick nearest cd=120 (logistic, train/test split) plus `|corr|`, then 5-fold-CV logistic and gradient-boosting multivariate AUC, benchmarked against raw UP mid price. Reports features with AUC > 0.55 and the best combined AUC.
- `sim_order_flow.py`: reconstructs UP/DN book series, computes spread, momentum, OU-reversion, and depth-imbalance features at cd in {180,120,90,60}, reports OOS AUC per feature, and runs mean-rev vs momentum head-to-head WR/EV on cheap (<= $0.50) entries.
- `sim_meanrev_window.py`: real-time first-trigger scan over the full entry-to-cd16 window. In UP_mid in [0.40,0.60], fade the side that fell over the last 20s. Reports fade WR and EV/$1 by cd-window and move threshold (0.8c/1.2c/1.6c/2.0c), with a fixed-cd comparison.

## Requirements

Python 3.9+. `sim_predictive_power.py` needs `numpy` and `scikit-learn` (gradient boosting degrades gracefully if scikit-learn is missing). The other two are pure stdlib.

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy scikit-learn
```

## Usage

```bash
source venv/bin/activate

# per-feature OOS AUC at cd~120 (optional arg = max log files, default 500)
python3 sim_predictive_power.py 500

# order-flow direction + mean-rev vs momentum cheap-entry study
python3 sim_order_flow.py

# coin-flip-zone mean-reversion fade across the full window
python3 sim_meanrev_window.py
```

## Data

Reads `./data` from the private polymarket-data repo; path constants are at the top of each file, point them at your local checkout. `sim_predictive_power.py` and `sim_order_flow.py` parse `race_test_btc-updown-5m-*.log` replay logs from `$DATA_DIR/quant_bots_logs_replay/`. `sim_meanrev_window.py` loads `$DATA_DIR/data/market_panel.json` and writes `.meanrev_window_summary.json`. No `.pkl` model files are loaded. All `.json`/`.jsonl`/`.log` data is git-ignored and must be supplied locally.

Read-only; no credentials or network access required.
