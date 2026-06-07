# predictive-signal-auc-research

Leak-free predictive-signal research for **Polymarket BTC up/down 5-minute binary markets**: mean-reversion fade windows, order-flow direction, and per-feature out-of-sample AUC studies. Pure offline backtests — nothing here trades or touches the live bot.

## Why it exists
Before any signal is wired into the live trading engine, it has to clear an honest, leak-free out-of-sample bar. These three studies measure whether candidate microstructure/momentum/mean-reversion signals actually predict the market winner — and at *which* countdown second the edge lives — so dead signals never reach production.

## What's inside

| Script | Question it answers | Method |
| --- | --- | --- |
| `sim_predictive_power.py` | Which math signal predicts the UP/DN winner at cd≈120s? | Per-feature univariate OOS AUC (logistic, train/test split, multi-seed) + `|corr|`, plus 5-fold-CV logistic & gradient-boosting multivariate AUC, benchmarked against the efficient-market reference (raw UP mid price). |
| `sim_order_flow.py` | Is order-flow/microstructure a direction predictor, and is mean-reversion or momentum the better cheap-entry rule? | Reconstructs UP/DN book series, computes spread/momentum/OU-reversion/depth-imbalance features at cd∈{180,120,90,60}, reports OOS AUC per feature (Mann-Whitney) and head-to-head mean-rev vs momentum WR/EV on cheap (≤$0.50) entries. |
| `sim_meanrev_window.py` | Does fading a 20s mid-move in the coin-flip zone pay across the full entry→cd16 window? | Real-time first-trigger scan (cd high→low): in UP_mid∈[0.40,0.60], fade the side that fell over 20s; reports fade WR / EV-per-$1 by cd-window and move threshold, with a fixed-cd comparison. |

All three are strictly **no-leak**: features come only from ticks at/before the decision point, the winner label is attached *after* extraction, and `sim_meanrev_window.py` uses first-trigger semantics (no future-aware sorting/dedup).

## Requirements
- Python 3.9+
- `numpy`, `scikit-learn` — required by `sim_predictive_power.py` (gradient boosting degrades gracefully if unavailable). The other two scripts are pure stdlib.
- No wallet, key, or network access — these are read-only backtests.

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy scikit-learn
```

## Usage
```bash
source venv/bin/activate

# 1) Per-feature OOS AUC at cd~120 (optional arg = max log files, default 500)
python3 sim_predictive_power.py 500

# 2) Order-flow direction + mean-rev vs momentum cheap-entry study
python3 sim_order_flow.py

# 3) Coin-flip-zone mean-reversion fade across the full window
python3 sim_meanrev_window.py
```

## Data
These scripts read inputs from the private **polymarket-data** repo (paths are currently hard-coded near the top of each file — point a `DATA_DIR` / the path constants there at your local checkout):
- `sim_predictive_power.py` and `sim_order_flow.py` parse `race_test_btc-updown-5m-*.log` replay logs from `quant_bots_logs_replay/`.
- `sim_meanrev_window.py` loads `data/market_panel.json` and writes a machine-readable `.meanrev_window_summary.json` for the orchestrator.

No model files (`.pkl`) are loaded. All `*.json`/`*.jsonl`/`*.log` data is git-ignored and must be supplied locally.

> Private research software. No warranty; trades/handles real funds at your own risk.
