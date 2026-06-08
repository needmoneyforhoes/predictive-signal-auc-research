#!/usr/bin/env python3
"""
sim_meanrev_window.py

Tests the coin-flip-zone SHORT-HORIZON MEAN-REVERSION FADE rule across the FULL
realistic entry->cd15 window, using REAL-TIME FIRST-TRIGGER semantics.

RULE (per market, scan ticks cd HIGH -> LOW, i.e. chronological):
  At the first tick where:
      UP_mid in [0.40, 0.60]              (coin-flip / "mid" zone)
      AND |20s mid move| > threshold      (something just moved)
  -> FIRE the side that FELL over the last 20s (fade / mean-revert).
       - if UP_mid ROSE (move>0)  -> UP rose, DN fell  -> FADE = buy DN
       - if UP_mid FELL (move<0)  -> UP fell           -> FADE = buy UP
  Buy at that side's ASK at the fire tick.
  Score with the market winner at settle (winner used ONLY for PnL, never for the
  decision). PnL per $1 staked = (1/entry - 1) if faded side wins else -1.

NO LEAK:
  - 20s mid move at cd=K is computed ONLY from up_mid at cd=K and the up_mid as of
    cd=K+20 (i.e. 20s EARLIER, which is a higher cd because cd counts down). We
    rebuild this ourselves from the tick stream rather than trusting any precomputed
    field, and we only ever look at ticks with cd >= K.
  - First-trigger: we stop at the FIRST qualifying tick scanning high->low.
  - Last 15s excluded: we never fire at cd <= 15 (unreliable fills).

Reports fade win-rate + EV/$1 for cd-windows:
  full(entry..16), 360-180, 180-120, 120-60, 60-16
for move thresholds: 0.8c, 1.2c, 1.6c, 2.0c.

Also reports a FIXED-CD comparison (fire at a single fixed cd snapshot) to show
whether first-trigger changes where the edge lives.
"""
import json
import sys
from collections import defaultdict

PANEL_PATH = "./data/market_panel.json"

MID_LO, MID_HI = 0.40, 0.60
CD_FLOOR = 16          # never fire at cd <= 15 (last 15s unreliable to fill)
LOOKBACK = 20          # 20-second horizon
THRESHOLDS = [0.008, 0.012, 0.016, 0.020]   # 0.8c, 1.2c, 1.6c, 2.0c

# cd-window buckets: (label, hi_inclusive, lo_inclusive)  scored by the fire cd
WINDOWS = [
    ("full(entry..16)", 10_000, CD_FLOOR),
    ("360-180",            360,     181),
    ("180-120",            180,     121),
    ("120-60",             120,      61),
    ("60-16",               60, CD_FLOOR),
]


def load_panel(path=PANEL_PATH):
    with open(path) as f:
        return json.load(f)


def build_cd_series(ticks):
    """
    Build a clean chronological list of (cd, up_mid, up_ask, dn_ask) using LAST-WINS
    per cd second (the latest event observed at that cd second is the freshest state).
    Returns list sorted by cd DESCENDING (chronological) and a cd->up_mid map for the
    20s-earlier lookup.

    LAST-WINS rationale: within a single cd second there can be several book events;
    the last one is the most recent state at that second. The 20s-earlier reference
    therefore uses the freshest state available as of cd+20, which is strictly in the
    past relative to the fire tick -> no leak.
    """
    cd_to_mid = {}
    cd_to_row = {}
    for t in ticks:
        cd = t["cd"]
        um = t.get("up_mid")
        if um is None:
            continue
        cd_to_mid[cd] = um  # last-wins
        cd_to_row[cd] = t   # last-wins
    # chronological order = cd descending
    ordered_cds = sorted(cd_to_row.keys(), reverse=True)
    series = []
    for cd in ordered_cds:
        t = cd_to_row[cd]
        series.append((cd, t.get("up_mid"), t.get("up_ask"), t.get("dn_ask")))
    return series, cd_to_mid


def mid_move_20s(cd, up_mid_now, cd_to_mid):
    """
    20s mid move = up_mid(now) - up_mid(as of cd+20).  Leak-free: cd+20 is earlier.
    Falls back to the nearest available cd in [cd+18, cd+22] if exact cd+20 missing,
    then [cd+15, cd+25]. Returns None if no reference within the horizon band.
    """
    for ref in (cd + LOOKBACK,):
        if ref in cd_to_mid:
            return up_mid_now - cd_to_mid[ref]
    # tolerant fallback: nearest higher cd within +/-2 of the 20s mark
    for delta in (1, 2):
        for ref in (cd + LOOKBACK + delta, cd + LOOKBACK - delta):
            if ref in cd_to_mid:
                return up_mid_now - cd_to_mid[ref]
    # wider fallback within +/-5
    best = None
    best_d = 99
    for ref in range(cd + LOOKBACK - 5, cd + LOOKBACK + 6):
        if ref in cd_to_mid:
            d = abs(ref - (cd + LOOKBACK))
            if d < best_d:
                best_d = d
                best = ref
    if best is not None:
        return up_mid_now - cd_to_mid[best]
    return None


def first_trigger_fire(market, threshold):
    """
    Scan ticks cd HIGH->LOW. Return the FIRST qualifying fire as a dict, or None.
    Leak-free: at fire cd K we only consult up_mid at cd>=K.
    """
    series, cd_to_mid = build_cd_series(market["ticks"])
    for cd, up_mid, up_ask, dn_ask in series:
        if cd <= CD_FLOOR:
            break  # everything below floor is unfireable; stop (series is descending)
        if up_mid is None or not (MID_LO <= up_mid <= MID_HI):
            continue
        move = mid_move_20s(cd, up_mid, cd_to_mid)
        if move is None or abs(move) <= threshold:
            continue
        # qualifying tick -> FADE the side that fell
        if move > 0:
            # UP rose => DN fell => fade buys DN
            side = "DN"
            entry = dn_ask
        else:
            # UP fell => fade buys UP
            side = "UP"
            entry = up_ask
        if entry is None or entry <= 0 or entry >= 1.0:
            continue  # unusable price; keep scanning (rare)
        return {"cd": cd, "side": side, "entry": entry, "move": move}
    return None


def fixed_cd_fire(market, threshold, target_cd, tol=10):
    """
    FIXED-CD comparison: take the tick closest to target_cd (within +/-tol, cd>target
    preferred for no-leak determinism) and apply the same fade rule once. Returns
    fire dict or None. Used only to contrast with first-trigger.
    """
    series, cd_to_mid = build_cd_series(market["ticks"])
    # pick the tick whose cd is closest to target_cd, not below CD_FLOOR
    cand = None
    cand_d = 999
    for cd, up_mid, up_ask, dn_ask in series:
        if cd <= CD_FLOOR:
            continue
        d = abs(cd - target_cd)
        if d < cand_d:
            cand_d = d
            cand = (cd, up_mid, up_ask, dn_ask)
    if cand is None or cand_d > tol:
        return None
    cd, up_mid, up_ask, dn_ask = cand
    if up_mid is None or not (MID_LO <= up_mid <= MID_HI):
        return None
    move = mid_move_20s(cd, up_mid, cd_to_mid)
    if move is None or abs(move) <= threshold:
        return None
    if move > 0:
        side, entry = "DN", dn_ask
    else:
        side, entry = "UP", up_ask
    if entry is None or entry <= 0 or entry >= 1.0:
        return None
    return {"cd": cd, "side": side, "entry": entry, "move": move}


def score(fire, winner):
    """PnL per $1 staked. Win -> (1/entry - 1); loss -> -1."""
    if fire["side"] == winner:
        return (1.0 / fire["entry"]) - 1.0
    return -1.0


def window_of(cd):
    for label, hi, lo in WINDOWS:
        if lo <= cd <= hi:
            yield label


def run_first_trigger(panel, threshold):
    """Return dict label -> stats for first-trigger over all markets."""
    buckets = {label: {"n": 0, "wins": 0, "pnl": 0.0, "cost": 0.0} for label, _, _ in WINDOWS}
    for m in panel:
        fire = first_trigger_fire(m, threshold)
        if fire is None:
            continue
        pnl = score(fire, m["winner"])
        won = 1 if fire["side"] == m["winner"] else 0
        for label in window_of(fire["cd"]):
            b = buckets[label]
            b["n"] += 1
            b["wins"] += won
            b["pnl"] += pnl       # PnL per $1 staked
            b["cost"] += 1.0      # $1 staked per fire
    out = {}
    for label, b in buckets.items():
        n = b["n"]
        out[label] = {
            "n": n,
            "wr": (b["wins"] / n) if n else 0.0,
            "ev_per_dollar": (b["pnl"] / b["cost"]) if b["cost"] else 0.0,
            "total_pnl": b["pnl"],
        }
    return out


def run_fixed_cd(panel, threshold, target_cd):
    n = wins = 0
    pnl = 0.0
    for m in panel:
        fire = fixed_cd_fire(m, threshold, target_cd)
        if fire is None:
            continue
        n += 1
        if fire["side"] == m["winner"]:
            wins += 1
        pnl += score(fire, m["winner"])
    return {
        "n": n,
        "wr": (wins / n) if n else 0.0,
        "ev_per_dollar": (pnl / n) if n else 0.0,
        "total_pnl": pnl,
    }


def main():
    panel = load_panel()
    print(f"Loaded {len(panel)} markets "
          f"({sum(1 for m in panel if m['winner']=='UP')} UP / "
          f"{sum(1 for m in panel if m['winner']=='DN')} DN)\n")
    print("RULE: coin-flip zone [0.40,0.60], FADE the side that FELL over 20s, "
          "first-trigger cd high->low, no fire at cd<=15.\n")

    # ---------------- FIRST-TRIGGER (the realistic semantics) ----------------
    print("=" * 92)
    print("FIRST-TRIGGER (real-time, leak-free) — fade WR / EV-per-$1 / n / totalPnL")
    print("=" * 92)
    hdr = f"{'thresh':>7} | " + " | ".join(f"{lab:^28}" for lab, _, _ in WINDOWS)
    for thr in THRESHOLDS:
        res = run_first_trigger(panel, thr)
        print(f"\n-- move-threshold = {thr*100:.1f}c " + "-" * 60)
        for label, _, _ in WINDOWS:
            r = res[label]
            print(f"   {label:<16} n={r['n']:>4}  WR={r['wr']*100:5.1f}%  "
                  f"EV/$1={r['ev_per_dollar']:+.4f}  totalPnL={r['total_pnl']:+8.2f}")

    # ---------------- FIXED-CD comparison ----------------
    print("\n" + "=" * 92)
    print("FIXED-CD SNAPSHOT (single cd, same fade rule) — to compare vs first-trigger")
    print("=" * 92)
    fixed_targets = [300, 240, 180, 150, 120, 90, 60, 40]
    for thr in THRESHOLDS:
        print(f"\n-- move-threshold = {thr*100:.1f}c " + "-" * 60)
        for tcd in fixed_targets:
            r = run_fixed_cd(panel, thr, tcd)
            print(f"   cd~{tcd:<4} n={r['n']:>4}  WR={r['wr']*100:5.1f}%  "
                  f"EV/$1={r['ev_per_dollar']:+.4f}  totalPnL={r['total_pnl']:+8.2f}")

    # ---------------- emit machine-readable summary for the orchestrator ----------------
    summary = {}
    for thr in THRESHOLDS:
        summary[f"{thr*100:.1f}c"] = run_first_trigger(panel, thr)
    with open("./data/.meanrev_window_summary.json", "w") as f:
        json.dump({"n_markets": len(panel), "first_trigger": summary}, f, indent=2)
    print("\n[wrote ./.meanrev_window_summary.json]")


if __name__ == "__main__":
    main()
