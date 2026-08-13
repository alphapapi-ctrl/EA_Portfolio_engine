"""
walk_forward_momentum_cost.py
=============================
Walk-forward validation of momentum WITH the streak-cost eligibility filter
(robots whose current losing streak has cost >= $X are excluded from
selection until it ends).

Calibration (pre-2023 only): grid of lookback x dollar threshold, best
calibration Sharpe wins. Test: frozen on 2023+. Result row is appended to
runs/_walk_forward/test_window_comparison.csv.
"""

import os

import pandas as pd

from portfolio_sim import load_timeline, simulate, REGIMES

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
WF_DIR     = os.path.join(ENGINE_DIR, 'runs', '_walk_forward')

TIMELINE     = 'main_pool'
TEST_START   = '2023-01-01'
N_SLOTS      = 10
GROSS        = 10.0
REVIEW_EVERY = 5
CALIB_WARMUP = 63
LOOKBACKS    = [21, 42, 63, 126, 189, 252]
DOLLAR_GRID  = [1000, 1500, 2000, 3000]


def make_regime(candidates, lb, dollar):
    return REGIMES['momentum'](candidates, GROSS, N_SLOTS, lookback=lb,
                               metric='sharpe', streak_dollar_limit=dollar)


def main():
    daily, _ = load_timeline(TIMELINE)
    candidates = list(daily.columns)
    split = int((daily.index < TEST_START).sum())
    calib = daily.iloc[:split]
    print(f"Calibration to {daily.index[split-1]:%Y-%m-%d}, "
          f"test from {daily.index[split]:%Y-%m-%d}")
    print(f"Calibrating {len(LOOKBACKS)}x{len(DOLLAR_GRID)} grid ...")

    best = None
    for lb in LOOKBACKS:
        for dollar in DOLLAR_GRID:
            s = simulate(calib, make_regime(candidates, lb, dollar),
                         review_every=REVIEW_EVERY, warmup=CALIB_WARMUP)['summary']
            if best is None or s['sharpe'] > best[2]:
                best = (lb, dollar, s['sharpe'])
        print(f"  lookback {lb} done", flush=True)
    lb, dollar, calib_sharpe = best
    print(f"Best on calibration: lookback={lb}, ${dollar} (sharpe {calib_sharpe:.2f})")

    res = simulate(daily, make_regime(candidates, lb, dollar),
                   review_every=REVIEW_EVERY, warmup=split)
    s = res['summary']
    row = {'run': f'momentum + streak-cost (calib lb={lb}, ${dollar})',
           **{k: s[k] for k in ['net_profit', 'ann_return_pct', 'ann_vol_pct',
                                'sharpe', 'max_dd', 'max_dd_pct', 'calmar',
                                'turnover_units', 'events']}}
    print("\nTest window result:")
    print(pd.Series(row).to_string())

    cmp_path = os.path.join(WF_DIR, 'test_window_comparison.csv')
    table = pd.read_csv(cmp_path)
    table = table[table['run'] != row['run']]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table = table.sort_values('sharpe', ascending=False)
    table.to_csv(cmp_path, index=False)
    res['equity'].to_csv(os.path.join(WF_DIR, 'momentum_streak_cost_equity.csv'))
    print(f"\nAppended to {cmp_path}")
    print(table[['run', 'sharpe', 'ann_return_pct', 'max_dd_pct',
                 'turnover_units']].to_string(index=False))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--review-every', type=int, default=REVIEW_EVERY,
                    help='Review cadence in trading days [5]')
    a = ap.parse_args()
    REVIEW_EVERY = a.review_every
    if a.review_every != 5:
        WF_DIR = WF_DIR + f'_r{a.review_every}'
    main()
