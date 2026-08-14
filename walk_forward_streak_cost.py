"""
walk_forward_streak_cost.py
===========================
Walk-forward validation of the streak-cost-only rules regime.

Calibration (data before 2023 only): pick the streak-cost threshold with the
best calibration Sharpe. Test: run frozen on 2023+. The result row is appended
to runs/_walk_forward/test_window_comparison.csv so it appears alongside the
original walk-forward table in the app.
"""

import os

import pandas as pd

from portfolio_sim import load_timeline, simulate, ea_stats, rank_metric, REGIMES

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
WF_DIR     = os.path.join(ENGINE_DIR, 'runs', '_walk_forward')

TIMELINE     = 'main_pool'
TEST_START   = '2023-01-01'
N_SLOTS      = 10
GROSS        = 10.0
REVIEW_EVERY = 5
CALIB_WARMUP = 63
DOLLAR_GRID  = [1000, 1500, 2000, 2500, 3000, 4000]


def make_regime(portfolio, candidates, dollar):
    return REGIMES['rules'](
        list(portfolio), candidates, GROSS, lookback=63, metric='sharpe',
        loss_streak_limit=None, ea_dd_limit_pct=None,
        streak_dollar_limit=dollar, corr_cap=0.7, cooldown_days=21)


def main():
    daily, _ = load_timeline(TIMELINE)
    candidates = list(daily.columns)
    split = int((daily.index < TEST_START).sum())
    calib = daily.iloc[:split]
    print(f"Calibration to {daily.index[split-1]:%Y-%m-%d}, "
          f"test from {daily.index[split]:%Y-%m-%d}")

    calib_stats = ea_stats(calib)
    portfolio   = rank_metric(calib_stats, 'sharpe').nlargest(N_SLOTS).index.tolist()

    best = None
    for dollar in DOLLAR_GRID:
        s = simulate(calib, make_regime(portfolio, candidates, dollar),
                     review_every=REVIEW_EVERY, warmup=CALIB_WARMUP)['summary']
        print(f"  calib ${dollar}: sharpe {s['sharpe']:.2f}")
        if best is None or s['sharpe'] > best[1]:
            best = (dollar, s['sharpe'])
    dollar, calib_sharpe = best
    print(f"Best on calibration: ${dollar} (sharpe {calib_sharpe:.2f})")

    res = simulate(daily, make_regime(portfolio, candidates, dollar),
                   review_every=REVIEW_EVERY, warmup=split)
    s = res['summary']
    row = {'run': f'rules streak-cost only (calib ${dollar})',
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
    res['equity'].to_csv(os.path.join(WF_DIR, 'rules_streak_cost_equity.csv'))
    print(f"\nAppended to {cmp_path}")
    print(table[['run', 'sharpe', 'ann_return_pct', 'max_dd_pct']].to_string(index=False))


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
        WF_DIR = WF_DIR + f'_r{a.review_every}'
    main()
