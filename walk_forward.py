"""
walk_forward.py
===============
EA Portfolio Engine — walk-forward validation.

Calibration window: everything strictly before TEST_START.
  - initial portfolio = top-10 by Sharpe over the calibration window
  - rules parameters  = best Sharpe cell of the loss_streak x dd_limit grid,
    with the grid evaluated ONLY on calibration data
  - momentum lookback = best Sharpe lookback, evaluated ONLY on calibration data

Test window: TEST_START onwards. All regimes run frozen — no re-selection of
parameters. Regimes still make their normal day-to-day decisions (that's the
strategy), but the *knobs* were chosen without seeing any test data.

Benchmarks on the test window: equal weight, static top-10 (ranked on calib),
random rotation averaged over 10 seeds.
"""

import os
import json

import numpy as np
import pandas as pd

from portfolio_sim import load_timeline, simulate, ea_stats, rank_metric, REGIMES

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(ENGINE_DIR, 'runs', '_walk_forward')

TIMELINE     = 'main_pool'
TEST_START   = '2023-01-01'
N_SLOTS      = 10
GROSS        = 10.0
REVIEW_EVERY = 5
CALIB_WARMUP = 63


def summarize(name, s):
    return {'run': name, **{k: s[k] for k in
            ['net_profit', 'ann_return_pct', 'ann_vol_pct', 'sharpe',
             'max_dd', 'max_dd_pct', 'calmar', 'turnover_units', 'events']}}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    daily, _ = load_timeline(TIMELINE)
    candidates = list(daily.columns)

    split = int((daily.index < TEST_START).sum())
    calib = daily.iloc[:split]
    print(f"Timeline {TIMELINE}: {len(daily)} days total")
    print(f"Calibration: {daily.index[0]:%Y-%m-%d} -> {daily.index[split-1]:%Y-%m-%d} ({split} days)")
    print(f"Test       : {daily.index[split]:%Y-%m-%d} -> {daily.index[-1]:%Y-%m-%d} ({len(daily)-split} days)")

    # ── Calibration: portfolio selection ─────────────────────────────────
    calib_stats = ea_stats(calib)
    portfolio   = rank_metric(calib_stats, 'sharpe').nlargest(N_SLOTS).index.tolist()
    print(f"\nPortfolio selected on calibration window:")
    for ea in portfolio:
        print(f"  {ea}")

    # ── Calibration: rules grid ──────────────────────────────────────────
    print("\nCalibrating rules grid on calibration window ...")
    best = None
    for st in [3, 4, 5, 6, 7, None]:
        for dd in [1.5, 2.0, 2.5, 3.0, 4.0, None]:
            if st is None and dd is None:
                continue
            regime = REGIMES['rules'](
                list(portfolio), candidates, GROSS, lookback=63, metric='sharpe',
                loss_streak_limit=st, ea_dd_limit_pct=dd,
                corr_cap=0.7, cooldown_days=21)
            s = simulate(calib, regime, review_every=REVIEW_EVERY,
                         warmup=CALIB_WARMUP)['summary']
            if best is None or s['sharpe'] > best[2]:
                best = (st, dd, s['sharpe'])
    best_streak, best_dd, best_calib_sharpe = best
    print(f"  Best on calib: loss_streak={best_streak}, dd_limit={best_dd} "
          f"(calib sharpe {best_calib_sharpe:.2f})")

    # ── Calibration: momentum lookback ───────────────────────────────────
    print("Calibrating momentum lookback on calibration window ...")
    best_lb = None
    for lb in [21, 42, 63, 126, 189, 252]:
        regime = REGIMES['momentum'](candidates, GROSS, N_SLOTS,
                                     lookback=lb, metric='sharpe')
        s = simulate(calib, regime, review_every=REVIEW_EVERY,
                     warmup=CALIB_WARMUP)['summary']
        if best_lb is None or s['sharpe'] > best_lb[1]:
            best_lb = (lb, s['sharpe'])
    lb, best_lb_sharpe = best_lb
    print(f"  Best on calib: lookback={lb} (calib sharpe {best_lb_sharpe:.2f})")

    # ── Test window: frozen regimes ──────────────────────────────────────
    # warmup=split means: flat through the calibration period, first review
    # at the first test day using full calibration history — and summary
    # stats are computed on the test window only.
    print("\nRunning frozen regimes on test window ...")
    rows = []

    regime = REGIMES['rules'](
        list(portfolio), candidates, GROSS, lookback=63, metric='sharpe',
        loss_streak_limit=best_streak, ea_dd_limit_pct=best_dd,
        corr_cap=0.7, cooldown_days=21)
    res = simulate(daily, regime, review_every=REVIEW_EVERY, warmup=split)
    rows.append(summarize('rules (calib params)', res['summary']))
    res['equity'].to_csv(os.path.join(OUT_DIR, 'rules_equity.csv'))
    pd.DataFrame(res['events']).to_csv(os.path.join(OUT_DIR, 'rules_events.csv'), index=False)

    regime = REGIMES['momentum'](candidates, GROSS, N_SLOTS,
                                 lookback=lb, metric='sharpe')
    res = simulate(daily, regime, review_every=REVIEW_EVERY, warmup=split)
    rows.append(summarize(f'momentum (calib lb={lb})', res['summary']))

    regime = REGIMES['equal_weight'](candidates, GROSS)
    res = simulate(daily, regime, review_every=REVIEW_EVERY, warmup=split)
    rows.append(summarize('equal weight (all)', res['summary']))

    regime = REGIMES['static_topn'](candidates, GROSS, N_SLOTS, metric='sharpe')
    res = simulate(daily, regime, review_every=REVIEW_EVERY, warmup=split)
    rows.append(summarize('static top10 (calib rank)', res['summary']))

    rand = []
    for seed in range(1, 11):
        regime = REGIMES['random'](candidates, GROSS, N_SLOTS, seed=seed)
        s = simulate(daily, regime, review_every=REVIEW_EVERY, warmup=split)['summary']
        rand.append(s)
    rand_df = pd.DataFrame(rand)
    rows.append({'run': 'random x10 (mean)',
                 **{k: round(float(rand_df[k].mean()), 2) for k in
                    ['net_profit', 'ann_return_pct', 'ann_vol_pct', 'sharpe',
                     'max_dd', 'max_dd_pct', 'calmar', 'turnover_units', 'events']}})

    table = pd.DataFrame(rows).set_index('run').sort_values('sharpe', ascending=False)
    table.to_csv(os.path.join(OUT_DIR, 'test_window_comparison.csv'))
    print("\n=== TEST WINDOW (out-of-sample for all knobs) ===")
    print(table.to_string())

    with open(os.path.join(OUT_DIR, 'calibration.json'), 'w') as f:
        json.dump({'test_start': TEST_START,
                   'portfolio': portfolio,
                   'rules': {'loss_streak_limit': best_streak,
                             'ea_dd_limit_pct': best_dd,
                             'calib_sharpe': best_calib_sharpe},
                   'momentum_lookback': lb}, f, indent=2)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--review-every', type=int, default=REVIEW_EVERY,
                    help='Review cadence in trading days [5]')
    ap.add_argument('--timeline', default=TIMELINE)
    a = ap.parse_args()
    TIMELINE = a.timeline
    REVIEW_EVERY = a.review_every
    if a.review_every != 5:
        OUT_DIR = OUT_DIR + f'_r{a.review_every}'
    main()
