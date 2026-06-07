#!/usr/bin/env python3
"""
sim_predictive_power.py

Empirically test which MATHEMATICAL SIGNALS predict the winner (UP vs DN) of
Polymarket BTC up/down 5-minute binary markets, using real replay logs.

Methodology (LEAK-FREE):
  * Decision point is FIXED at cd ~= 120s (tick nearest cd=120, within [100,140]).
  * Only ticks AT/BEFORE cd=120 are ever read. No future ticks, no winner-derived
    sorting/dedup. Label (winner) is attached AFTER feature extraction.
  * Univariate predictive power: per-feature, fit a logistic on a TRAIN split and
    score AUC on a held-out TEST split (out-of-sample). Also abs Pearson corr (full).
  * Multivariate: 5-fold CV logistic + (if avail) gradient boosting; mean OOS AUC.

Run:
  source venv/bin/activate
  python3 sim_predictive_power.py [max_logs]
"""
import os
import re
import sys
import glob
import math
import numpy as np

LOG_DIR = "/home/polybot/polymarket-bot/quant_bots_logs_replay"
GLOB = os.path.join(LOG_DIR, "race_test_btc-updown-5m-*.log")

DECISION_CD = 120          # target seconds remaining
CD_WINDOW = (100, 140)     # accept nearest tick within this band
SLOPE_LOOKBACK_S = 30      # trajectory slope over last 30s before decision

# ---- regexes -----------------------------------------------------------------
RE_PRICE = re.compile(
    r"\[price\]\s+(\d+)s\s+\|\s+UP bid=([\d.]+) ask=([\d.]+)\s+\|\s+"
    r"DN bid=([\d.]+) ask=([\d.]+)\s+\|\s+ask_sum=([\d.]+) bid_sum=([\d.]+)"
)
RE_SIGNAL_CD = re.compile(r"\[signal\]\s+(\d+)s")
RE_WINNER = re.compile(r"winner=([A-Za-z]+)")


def _kv(line, key):
    """Extract float value for `key=...` token from a signal/price line."""
    m = re.search(re.escape(key) + r"=([+\-]?[\d.]+)", line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _crowd(line):
    m = re.search(r"crowd=([A-Za-z]+)", line)
    if not m:
        return None
    v = m.group(1)
    if v == "UP":
        return 1.0
    if v == "DN":
        return 0.0
    return None  # None/other


def parse_log(path):
    """
    Returns dict of features at the decision point, plus 'winner' (1=UP,0=DN),
    or None if the market can't be used (no cd~120 tick / unknown winner).
    Strictly only reads ticks with cd >= DECISION_CD's nearest band (we read all
    lines, but only KEEP those at/after the decision point for slope; final
    features come from the single nearest tick).
    """
    winner = None
    # price ticks: list of (cd, up_bid, up_ask, dn_bid, dn_ask, ask_sum, bid_sum)
    prices = []
    # signal ticks: list of (cd, line)  -- keep raw line, parse lazily
    signals = []

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if "winner=" in line and "RECAP" in line:
                m = RE_WINNER.search(line)
                if m:
                    winner = m.group(1)
                continue
            if "[price]" in line:
                m = RE_PRICE.search(line)
                if m:
                    cd = int(m.group(1))
                    prices.append((
                        cd,
                        float(m.group(2)), float(m.group(3)),
                        float(m.group(4)), float(m.group(5)),
                        float(m.group(6)), float(m.group(7)),
                    ))
                continue
            if "[signal]" in line:
                m = RE_SIGNAL_CD.search(line)
                if m:
                    signals.append((int(m.group(1)), line))
                continue

    if winner not in ("UP", "DN"):
        return None
    if not prices:
        return None

    # --- find price tick nearest cd=120 within window -------------------------
    cand = [p for p in prices if CD_WINDOW[0] <= p[0] <= CD_WINDOW[1]]
    if not cand:
        return None
    dpt = min(cand, key=lambda p: abs(p[0] - DECISION_CD))
    dcd = dpt[0]
    up_bid, up_ask, dn_bid, dn_ask, ask_sum, bid_sum = dpt[1:]

    # --- find signal tick nearest the SAME cd (only at/after decision cd) ------
    # (signals counted down; "at/after" decision means cd >= dcd, i.e. not future)
    sig_cand = [s for s in signals if s[0] >= dcd]
    sig_line = None
    if sig_cand:
        sig_line = min(sig_cand, key=lambda s: s[0] - dcd)[1]
    else:
        # fallback: nearest available signal at-or-before in time but allow small slack
        near = [s for s in signals if abs(s[0] - dcd) <= 5]
        if near:
            sig_line = min(near, key=lambda s: abs(s[0] - dcd))[1]

    feat = {}

    # --- price-derived features (all available AT decision point) -------------
    up_mid = (up_bid + up_ask) / 2.0
    dn_mid = (dn_bid + dn_ask) / 2.0
    up_spread = up_ask - up_bid
    dn_spread = dn_ask - dn_bid
    feat["UP_mid_price"] = up_mid
    feat["DN_mid_price"] = dn_mid
    feat["mid_diff"] = up_mid - dn_mid           # UP-DN mid skew
    feat["bid_sum"] = bid_sum
    feat["ask_sum"] = ask_sum
    feat["spread"] = up_spread + dn_spread       # total book spread
    feat["up_spread"] = up_spread
    feat["dn_spread"] = dn_spread
    # order-flow proxies derivable w/o explicit size:
    #   spread imbalance: tighter side = more liquidity/conviction
    feat["spread_imbalance"] = dn_spread - up_spread  # >0 => UP tighter (UP favored)
    #   bid/ask pressure: how much room above/below mid
    feat["up_bid_room"] = up_mid - up_bid
    feat["up_ask_room"] = up_ask - up_mid

    # --- signal-derived features ---------------------------------------------
    if sig_line is not None:
        conv = _kv(sig_line, "conv")
        vel = _kv(sig_line, "vel")
        ema_up = _kv(sig_line, "ema_up")
        ema_dn = _kv(sig_line, "ema_dn")
        d3u = _kv(sig_line, "Δ3s_up")
        d3d = _kv(sig_line, "Δ3s_dn")
        d10u = _kv(sig_line, "Δ10s_up")
        d10d = _kv(sig_line, "Δ10s_dn")
        crowd = _crowd(sig_line)
    else:
        conv = vel = ema_up = ema_dn = d3u = d3d = d10u = d10d = crowd = None

    feat["conv"] = conv
    feat["vel"] = vel
    feat["ema_up"] = ema_up
    feat["ema_dn"] = ema_dn
    feat["ema_diff"] = (ema_up - ema_dn) if (ema_up is not None and ema_dn is not None) else None
    feat["d3s_up"] = d3u
    feat["d3s_dn"] = d3d
    feat["d10s_up"] = d10u
    feat["d10s_dn"] = d10d
    feat["d3s_diff"] = (d3u - d3d) if (d3u is not None and d3d is not None) else None
    feat["d10s_diff"] = (d10u - d10d) if (d10u is not None and d10d is not None) else None
    feat["crowd"] = crowd

    # --- price TRAJECTORY slope over last 30s BEFORE cd=120 -------------------
    # "before cd=120" => higher cd values (earlier in time). Use ticks with
    # cd in [dcd, dcd+SLOPE_LOOKBACK_S]. Regress UP_mid vs time(seconds elapsed).
    # time elapsed = (max_cd - cd) so slope is per-second of real time. LEAK-FREE
    # (all these cd >= dcd, i.e. at/before the decision moment).
    traj = [p for p in prices if dcd <= p[0] <= dcd + SLOPE_LOOKBACK_S]
    slope = None
    if len(traj) >= 3:
        # x = real-time seconds = (dcd + lookback - cd) ascending ; y = up_mid
        xs = np.array([(SLOPE_LOOKBACK_S + dcd - p[0]) for p in traj], dtype=float)
        ys = np.array([(p[1] + p[2]) / 2.0 for p in traj], dtype=float)  # up mid
        if xs.std() > 1e-9:
            # slope of up_mid per second over the trailing 30s
            slope = float(np.polyfit(xs, ys, 1)[0])
    feat["up_mid_slope30"] = slope

    feat["winner"] = 1.0 if winner == "UP" else 0.0
    feat["_dcd"] = float(dcd)
    return feat


# ---- modeling ----------------------------------------------------------------
def univariate_auc(x, y, seed=42):
    """OOS AUC: fit univariate logistic on train split, score AUC on test split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    # need both classes present
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        xtr, xte, ytr, yte = train_test_split(
            x, y, test_size=0.35, random_state=seed, stratify=y
        )
    except ValueError:
        return float("nan")
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    # standardize using train stats only
    mu, sd = xtr.mean(), xtr.std()
    sd = sd if sd > 1e-9 else 1.0
    clf = LogisticRegression(max_iter=1000)
    clf.fit((xtr - mu) / sd, ytr)
    p = clf.predict_proba((xte - mu) / sd)[:, 1]
    auc = roc_auc_score(yte, p)
    # AUC is symmetric direction-wise; report >=0.5 (edge magnitude)
    return max(auc, 1.0 - auc)


def abs_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return abs(float(np.corrcoef(x, y)[0, 1]))


def main():
    max_logs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    files = sorted(glob.glob(GLOB))[:max_logs]
    print(f"[load] {len(files)} log files (cap {max_logs})")

    rows = []
    skipped = 0
    for f in files:
        try:
            r = parse_log(f)
        except Exception as e:
            r = None
        if r is None:
            skipped += 1
            continue
        rows.append(r)
    n = len(rows)
    print(f"[parse] usable markets: {n}  (skipped {skipped})")
    if n < 30:
        print("Not enough markets parsed.")
        return

    y = np.array([r["winner"] for r in rows], dtype=float)
    up_rate = y.mean()
    print(f"[label] UP rate = {up_rate:.3f}  (n_UP={int(y.sum())}, n_DN={int((1-y).sum())})")

    feature_names = [
        "conv", "vel", "ema_up", "ema_dn", "ema_diff",
        "d3s_up", "d3s_dn", "d10s_up", "d10s_dn", "d3s_diff", "d10s_diff",
        "crowd", "UP_mid_price", "DN_mid_price", "mid_diff",
        "bid_sum", "ask_sum", "spread", "up_spread", "dn_spread",
        "spread_imbalance", "up_bid_room", "up_ask_room", "up_mid_slope30",
    ]

    # ---- per-feature univariate OOS AUC + corr ------------------------------
    print("\n=== UNIVARIATE OUT-OF-SAMPLE PREDICTIVE POWER (cd~120) ===")
    print(f"{'feature':18s} {'n':>4s} {'OOS_AUC':>8s} {'|corr|':>7s}  edge")
    results = {}
    for fn in feature_names:
        vals = [r.get(fn) for r in rows]
        mask = np.array([v is not None and not (isinstance(v, float) and math.isnan(v)) for v in vals])
        if mask.sum() < 30:
            results[fn] = (mask.sum(), float("nan"), float("nan"))
            print(f"{fn:18s} {int(mask.sum()):4d} {'  n/a':>8s} {'n/a':>7s}  (too few)")
            continue
        xv = np.array([float(v) for v, m in zip(vals, mask) if m])
        yv = y[mask]
        # average AUC across a few seeds for stability
        aucs = [univariate_auc(xv, yv, seed=s) for s in (1, 7, 13, 42, 99)]
        aucs = [a for a in aucs if not math.isnan(a)]
        auc = float(np.mean(aucs)) if aucs else float("nan")
        c = abs_corr(xv, yv)
        results[fn] = (int(mask.sum()), auc, c)
        edge = "EDGE" if (not math.isnan(auc) and auc > 0.55) else ""
        print(f"{fn:18s} {int(mask.sum()):4d} {auc:8.3f} {c:7.3f}  {edge}")

    # ---- multivariate: build matrix, impute medians -------------------------
    print("\n=== MULTIVARIATE (5-fold CV, OOS AUC) ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = np.full((n, len(feature_names)), np.nan)
    for i, r in enumerate(rows):
        for j, fn in enumerate(feature_names):
            v = r.get(fn)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                X[i, j] = float(v)
    # median impute (train-fold medians applied via pipeline-friendly manual CV)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def cv_auc(make_model):
        aucs = []
        for tr, te in skf.split(X, y):
            med = np.nanmedian(X[tr], axis=0)
            med = np.where(np.isnan(med), 0.0, med)
            Xtr = np.where(np.isnan(X[tr]), med, X[tr])
            Xte = np.where(np.isnan(X[te]), med, X[te])
            mdl = make_model()
            mdl.fit(Xtr, y[tr])
            p = mdl.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        return float(np.mean(aucs)), float(np.std(aucs))

    logit_auc, logit_sd = cv_auc(
        lambda: Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(max_iter=2000, C=0.5))])
    )
    print(f"Logistic (all {len(feature_names)} feats):  OOS AUC = {logit_auc:.3f} ± {logit_sd:.3f}")

    gb_auc = float("nan")
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        gb_auc, gb_sd = cv_auc(
            lambda: GradientBoostingClassifier(
                n_estimators=120, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42)
        )
        print(f"GradientBoosting:                    OOS AUC = {gb_auc:.3f} ± {gb_sd:.3f}")
    except Exception as e:
        print(f"GradientBoosting unavailable: {e}")

    # baselines: always-UP, and the market's own implied prob (UP_mid_price)
    base_majority = max(up_rate, 1 - up_rate)
    print(f"\n[baseline] majority-class accuracy = {base_majority:.3f}")
    up_mid_vals = X[:, feature_names.index("UP_mid_price")]
    m2 = ~np.isnan(up_mid_vals)
    if m2.sum() > 30 and len(np.unique(y[m2])) == 2:
        mkt_auc = roc_auc_score(y[m2], up_mid_vals[m2])
        print(f"[baseline] market price (UP_mid) AUC = {mkt_auc:.3f}  "
              f"(this is the EFFICIENT-MARKET reference)")

    # ---- summary ------------------------------------------------------------
    print("\n=== FEATURES WITH AUC > 0.55 ===")
    edges = [(fn, results[fn][1], results[fn][2])
             for fn in feature_names
             if not math.isnan(results[fn][1]) and results[fn][1] > 0.55]
    edges.sort(key=lambda t: -t[1])
    if edges:
        for fn, a, c in edges:
            print(f"  {fn:18s} AUC={a:.3f}  |corr|={c:.3f}")
    else:
        print("  NONE. No single math signal beats AUC 0.55 at cd~120.")

    best_combo = max(
        [("logistic_all", logit_auc), ("gradient_boost", gb_auc)],
        key=lambda t: (t[1] if not math.isnan(t[1]) else -1),
    )
    print(f"\n[BEST COMBINED] {best_combo[0]} OOS AUC = {best_combo[1]:.3f}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
