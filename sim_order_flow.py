#!/usr/bin/env python3
"""
sim_order_flow.py

Deep empirical test of ORDER-FLOW / MICROSTRUCTURE math as a DIRECTION predictor
for Polymarket BTC up/down 5-minute binary markets.

For each replay log:
  - Reconstruct per-market time series of UP bid/ask, DN bid/ask from [price] lines.
  - Implied UP mid = (UP_bid + UP_ask)/2. (DN mid is ~ 1 - UP mid.)
  - At decision points cd in {180,120,90,60}, compute microstructure features:
       * spread (UP ask - UP bid) and spread dynamics (change over last ~20s)
       * momentum: rate of change of UP mid over a trailing window
       * mean-reversion (OU): deviation of UP mid from its short rolling mean (z-score-ish)
       * depth-imbalance proxy: ask_sum vs bid_sum dislocation, and bid_diff (UPbid-DNbid)
  - Label = market winner (UP=1, DN=0).

Then:
  - OOS AUC for each feature (predicting UP win) via simple 50/50 time-split.
  - Head-to-head MEAN-REVERSION rule vs MOMENTUM rule at each cd:
       compute win-rate and EV for CHEAP entries (entry price <= $0.50) under each.
  - Report at which cd the edge is largest (optimal entry timing).

WS-SAFETY: This is a pure OFFLINE backtest reading log files. Nothing here runs in
the bot. The signals tested (rolling mid mean, spread deltas, ask_sum/bid_sum
dislocation) are all O(1)-updatable from a small ring buffer -> if wired into the
bot they are WS-safe ONLY if computed incrementally in a worker thread, never
re-scanned inline per WS event.
"""

import os
import re
import glob
import math
import random

LOG_DIR = "./quant_bots_logs_replay"
MAX_FILES = 500
DECISION_CDS = [180, 120, 90, 60]
CHEAP_MAX = 0.50  # cheap entry threshold

# Regexes
RE_PRICE = re.compile(
    r"\[price\]\s+(\d+)s\s*\|\s*"
    r"UP bid=([0-9.]+) ask=([0-9.]+)\s*\|\s*"
    r"DN bid=([0-9.]+) ask=([0-9.]+)\s*\|\s*"
    r"ask_sum=([0-9.]+) bid_sum=([0-9.]+)"
)
RE_WINNER = re.compile(r"MARKET RECAP — \S+ \(winner=(UP|DN)\)")


def parse_log(path):
    """Return (ticks, winner) where ticks is list of dicts ordered by appearance.
    Each tick: cd, up_bid, up_ask, dn_bid, dn_ask, ask_sum, bid_sum, up_mid."""
    ticks = []
    winner = None
    try:
        with open(path, "r", errors="ignore") as fh:
            for line in fh:
                if "[price]" in line:
                    m = RE_PRICE.search(line)
                    if m:
                        cd = int(m.group(1))
                        up_bid = float(m.group(2)); up_ask = float(m.group(3))
                        dn_bid = float(m.group(4)); dn_ask = float(m.group(5))
                        ask_sum = float(m.group(6)); bid_sum = float(m.group(7))
                        up_mid = (up_bid + up_ask) / 2.0
                        ticks.append({
                            "cd": cd, "up_bid": up_bid, "up_ask": up_ask,
                            "dn_bid": dn_bid, "dn_ask": dn_ask,
                            "ask_sum": ask_sum, "bid_sum": bid_sum,
                            "up_mid": up_mid,
                        })
                elif winner is None and "MARKET RECAP" in line:
                    w = RE_WINNER.search(line)
                    if w:
                        winner = w.group(1)
    except (IOError, OSError):
        return None, None
    return ticks, winner


def tick_at_or_after(ticks, target_cd):
    """ticks are in chronological order => cd is DECREASING.
    Return the first tick with cd <= target_cd (i.e. the moment we cross the
    decision point), or None if market never reached that cd."""
    for t in ticks:
        if t["cd"] <= target_cd:
            return t
    return None


def window_mid_series(ticks, ref_cd, lookback_s):
    """Return list of up_mid for ticks with cd in [ref_cd, ref_cd+lookback_s],
    i.e. the trailing window BEFORE we crossed ref_cd (older time = larger cd).
    Ordered oldest->newest (descending cd -> we reverse to chronological)."""
    sel = [t["up_mid"] for t in ticks if ref_cd <= t["cd"] <= ref_cd + lookback_s]
    # ticks were chronological (cd descending), so sel is newest..oldest within the
    # slice ordering of the file; reverse to oldest->newest for momentum sign clarity
    return sel


def compute_features(ticks, cd):
    """Compute microstructure features at decision point cd. Return dict or None."""
    t0 = tick_at_or_after(ticks, cd)
    if t0 is None:
        return None
    ref_cd = t0["cd"]
    up_mid = t0["up_mid"]
    spread = t0["up_ask"] - t0["up_bid"]
    # depth-imbalance proxies
    asksum_disloc = t0["ask_sum"] - 1.0     # >0 means overround on asks
    bidsum_disloc = 1.0 - t0["bid_sum"]     # >0 means underround on bids
    sum_gap = t0["ask_sum"] - t0["bid_sum"] # total quoted spread across both sides
    bid_diff = t0["up_bid"] - t0["dn_bid"]  # crowd lean via bids
    ask_diff = t0["up_ask"] - t0["dn_ask"]

    # --- Momentum: change in up_mid over trailing ~20s window ---
    win20 = window_mid_series(ticks, ref_cd, 20)
    # win20 newest..oldest -> oldest is last element
    mom20 = None
    if len(win20) >= 2:
        newest = win20[0]
        oldest = win20[-1]
        mom20 = newest - oldest  # >0 = UP mid rising recently

    # --- Momentum over ~40s ---
    win40 = window_mid_series(ticks, ref_cd, 40)
    mom40 = None
    if len(win40) >= 2:
        mom40 = win40[0] - win40[-1]

    # --- Mean-reversion (OU): deviation of current mid from short rolling mean ---
    # rolling mean over trailing 30s; reversion signal = -(mid - mean) normalized
    win30 = window_mid_series(ticks, ref_cd, 30)
    ou_dev = None      # current - mean (positive = stretched UP, expect pullback DN)
    ou_z = None
    if len(win30) >= 3:
        mean30 = sum(win30) / len(win30)
        var = sum((x - mean30) ** 2 for x in win30) / len(win30)
        sd = math.sqrt(var) if var > 0 else 0.0
        ou_dev = up_mid - mean30
        ou_z = (up_mid - mean30) / sd if sd > 1e-6 else 0.0

    # --- Spread dynamics: change in spread over trailing 20s ---
    spread_chg = None
    sl = [t["up_ask"] - t["up_bid"] for t in ticks if ref_cd <= t["cd"] <= ref_cd + 20]
    if len(sl) >= 2:
        spread_chg = sl[0] - sl[-1]  # newest - oldest

    return {
        "ref_cd": ref_cd,
        "up_mid": up_mid,
        "up_ask": t0["up_ask"],
        "dn_ask": t0["dn_ask"],
        "spread": spread,
        "spread_chg": spread_chg,
        "mom20": mom20,
        "mom40": mom40,
        "ou_dev": ou_dev,
        "ou_z": ou_z,
        "asksum_disloc": asksum_disloc,
        "bidsum_disloc": bidsum_disloc,
        "sum_gap": sum_gap,
        "bid_diff": bid_diff,
        "ask_diff": ask_diff,
    }


def auc(scores, labels):
    """AUC = P(score(pos) > score(neg)). scores higher => predict label 1 (UP win)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    # rank-based (Mann-Whitney)
    allv = sorted([(s, 0) for s in neg] + [(s, 1) for s in pos], key=lambda x: x[0])
    # assign ranks with tie handling
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    sum_pos_ranks = sum(rk for rk, (_, y) in zip(ranks, allv) if y == 1)
    n_pos = len(pos); n_neg = len(neg)
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def main():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "race_test_btc-updown-5m-*.log")))
    files = files[:MAX_FILES]

    markets = []  # list of (features_by_cd, winner_label)
    n_no_winner = 0
    n_no_ticks = 0
    for path in files:
        ticks, winner = parse_log(path)
        if not ticks:
            n_no_ticks += 1
            continue
        if winner not in ("UP", "DN"):
            n_no_winner += 1
            continue
        label = 1 if winner == "UP" else 0
        feats = {}
        for cd in DECISION_CDS:
            f = compute_features(ticks, cd)
            if f is not None:
                feats[cd] = f
        markets.append((feats, label))

    n = len(markets)
    print(f"[info] files scanned={len(files)} usable_markets={n} "
          f"no_winner={n_no_winner} no_ticks={n_no_ticks}")
    up_rate = sum(1 for _, y in markets if y == 1) / n if n else 0
    print(f"[info] base UP-win rate = {up_rate:.3f}")

    # ---------------- OOS AUC per feature per cd ----------------
    # Time-split: first 50% train (not really fit, just direction-sign calibration),
    # second 50% test. For monotone features we don't fit params; AUC is computed on
    # the TEST half only. We orient each feature so higher => predict UP using the
    # TRAIN half sign, then evaluate AUC on TEST half (so AUC>0.5 is genuine OOS).
    split = n // 2
    train = markets[:split]
    test = markets[split:]

    feat_names = ["up_mid", "mom20", "mom40", "ou_dev", "ou_z", "spread",
                  "spread_chg", "asksum_disloc", "bidsum_disloc", "sum_gap",
                  "bid_diff", "ask_diff"]

    print("\n================ OOS AUC (predict UP win), TEST half ================")
    auc_results = {}  # (cd, feat) -> (auc, n)
    for cd in DECISION_CDS:
        print(f"\n--- cd={cd}s ---")
        for fn in feat_names:
            # orient on train
            tr_s = []; tr_y = []
            for feats, y in train:
                if cd in feats and feats[cd].get(fn) is not None:
                    tr_s.append(feats[cd][fn]); tr_y.append(y)
            te_s = []; te_y = []
            for feats, y in test:
                if cd in feats and feats[cd].get(fn) is not None:
                    te_s.append(feats[cd][fn]); te_y.append(y)
            if len(tr_s) < 10 or len(te_s) < 10:
                continue
            a_tr = auc(tr_s, tr_y)
            if a_tr is None:
                continue
            sign = 1.0 if a_tr >= 0.5 else -1.0
            te_oriented = [sign * v for v in te_s]
            a_te = auc(te_oriented, te_y)
            if a_te is None:
                continue
            auc_results[(cd, fn)] = (a_te, len(te_s), sign)
            flag = "***" if abs(a_te - 0.5) >= 0.07 else ("*" if abs(a_te - 0.5) >= 0.04 else "")
            print(f"  {fn:16s} OOS_AUC={a_te:.3f} (train_AUC={a_tr:.3f} sign={int(sign):+d}) n={len(te_s):4d} {flag}")

    # ---------------- MEAN-REVERSION vs MOMENTUM decision rules ----------------
    # Rule construction at each cd:
    #   MOMENTUM rule: trade in the direction of recent mid move.
    #       if mom20 > +thr -> buy UP at up_ask ; if mom20 < -thr -> buy DN at dn_ask
    #   MEAN-REVERSION rule: trade AGAINST a stretch from rolling mean.
    #       if ou_dev > +thr (UP stretched high) -> buy DN at dn_ask (expect pullback)
    #       if ou_dev < -thr (UP stretched low)  -> buy UP at up_ask (expect bounce)
    # Only count CHEAP entries: chosen side's ask <= CHEAP_MAX.
    # PnL per trade (binary, $1 stake notion): if side wins -> (1/price - 1); else -1.
    # EV reported as average PnL per $1 risked. Win-rate = fraction of correct-side.
    #
    # Thresholds chosen as small fixed values typical of this microstructure.
    MOM_THR = 0.01   # 1c mid move over 20s
    OU_THR = 0.012   # 1.2c deviation from rolling mean

    print("\n========== MEAN-REVERSION vs MOMENTUM (cheap entries <= $%.2f) ==========" % CHEAP_MAX)

    rule_summary = {}  # (cd, ruletype) -> dict

    def eval_rule(cd, ruletype):
        n_trades = 0; n_win = 0; pnl_sum = 0.0
        entry_prices = []
        for feats, y in markets:
            if cd not in feats:
                continue
            f = feats[cd]
            side = None; price = None
            if ruletype == "momentum":
                if f["mom20"] is None:
                    continue
                if f["mom20"] > MOM_THR:
                    side = "UP"; price = f["up_ask"]
                elif f["mom20"] < -MOM_THR:
                    side = "DN"; price = f["dn_ask"]
            elif ruletype == "meanrev":
                if f["ou_dev"] is None:
                    continue
                if f["ou_dev"] > OU_THR:
                    side = "DN"; price = f["dn_ask"]   # UP overshot -> fade to DN
                elif f["ou_dev"] < -OU_THR:
                    side = "UP"; price = f["up_ask"]   # UP undershot -> buy UP
            if side is None or price is None:
                continue
            if price > CHEAP_MAX or price <= 0.0 or price >= 1.0:
                continue
            n_trades += 1
            entry_prices.append(price)
            won = (side == "UP" and y == 1) or (side == "DN" and y == 0)
            if won:
                n_win += 1
                pnl_sum += (1.0 / price - 1.0)
            else:
                pnl_sum += -1.0
        if n_trades == 0:
            return None
        wr = n_win / n_trades
        ev = pnl_sum / n_trades
        avg_price = sum(entry_prices) / len(entry_prices)
        return {"n": n_trades, "wr": wr, "ev": ev, "avg_price": avg_price}

    for cd in DECISION_CDS:
        mom = eval_rule(cd, "momentum")
        mr = eval_rule(cd, "meanrev")
        rule_summary[(cd, "momentum")] = mom
        rule_summary[(cd, "meanrev")] = mr
        print(f"\n--- cd={cd}s ---")
        for label, r in (("MOMENTUM", mom), ("MEAN-REV", mr)):
            if r is None:
                print(f"  {label:9s}: no qualifying cheap trades")
            else:
                print(f"  {label:9s}: trades={r['n']:4d}  WR={r['wr']:.3f}  "
                      f"EV/$1={r['ev']:+.4f}  avg_entry=${r['avg_price']:.3f}")

    # ---------------- Verdict: best cd & approach ----------------
    print("\n================ SUMMARY: optimal entry timing ================")
    best = None
    for (cd, rt), r in rule_summary.items():
        if r is None or r["n"] < 15:
            continue
        # rank by EV (edge), require positive
        if best is None or r["ev"] > best[2]["ev"]:
            best = (cd, rt, r)
    if best:
        cd, rt, r = best
        print(f"BEST cheap-entry rule: {rt.upper()} @ cd={cd}s  "
              f"WR={r['wr']:.3f} EV/$1={r['ev']:+.4f} trades={r['n']}")
    else:
        print("No cheap-entry rule cleared the n>=15 bar.")

    # best AUC feature overall
    best_auc = None
    for (cd, fn), (a, nn, sgn) in auc_results.items():
        if nn < 20:
            continue
        edge = abs(a - 0.5)
        if best_auc is None or edge > best_auc[2]:
            best_auc = ((cd, fn), a, edge, sgn)
    if best_auc:
        (cd, fn), a, edge, sgn = best_auc
        print(f"BEST OOS-AUC feature: {fn} @ cd={cd}s  AUC={a:.3f} (sign={int(sgn):+d})")

    return markets, auc_results, rule_summary, up_rate


if __name__ == "__main__":
    main()
