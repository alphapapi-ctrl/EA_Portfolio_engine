"""
sweep_streak_cost.py
====================
Parameter sweep for the streak-cost benching rule:
grid of streak_dollar_limit x ea_dd_limit_pct (corr cap 0.7, no count-based
streak rule, so the cost rule's contribution is isolated). Same portfolio and
cadence as the original rules grid, for direct comparability.

Saves runs/_sweeps/streak_cost_grid.csv and prints Sharpe / max DD tables.
"""

import os

import pandas as pd

from portfolio_sim import load_timeline, simulate, ea_stats, pick_top, REGIMES

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(ENGINE_DIR, 'runs', '_sweeps')

TIMELINE     = 'main_pool'
N_SLOTS      = 10
GROSS        = 10.0
REVIEW_EVERY = 5
WARMUP       = 63

DOLLAR_LIMITS = [1000, 1500, 2000, 2500, 3000, 4000]
DD_LIMITS     = [1.5, 2.0, 2.5, 3.0, 4.0, None]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    daily, _ = load_timeline(TIMELINE)
    print(f"Timeline {TIMELINE}: {daily.shape[0]} days x {daily.shape[1]} EAs")
    candidates = list(daily.columns)
    stats      = ea_stats(daily.iloc[:WARMUP])
    portfolio  = pick_top(stats, 'sharpe', N_SLOTS)

    grid_sharpe = pd.DataFrame(index=[str(d) for d in DOLLAR_LIMITS],
                               columns=[str(d) for d in DD_LIMITS], dtype=float)
    grid_dd     = grid_sharpe.copy()
    records = []
    for dollar in DOLLAR_LIMITS:
        for dd in DD_LIMITS:
            regime = REGIMES['rules'](
                list(portfolio), candidates, GROSS,
                lookback=63, metric='sharpe',
                loss_streak_limit=None, streak_dollar_limit=dollar,
                ea_dd_limit_pct=dd, corr_cap=0.7, cooldown_days=21)
            s = simulate(daily, regime, review_every=REVIEW_EVERY,
                         warmup=WARMUP)['summary']
            grid_sharpe.loc[str(dollar), str(dd)] = s['sharpe']
            grid_dd.loc[str(dollar), str(dd)]     = s['max_dd_pct']
            records.append({'streak_dollar_limit': dollar,
                            'ea_dd_limit_pct': dd, **s})
        print(f"  dollar_limit={dollar} done", flush=True)

    pd.DataFrame(records).to_csv(os.path.join(OUT_DIR, 'streak_cost_grid.csv'),
                                 index=False)
    print("\nSharpe (rows=streak_dollar_limit, cols=ea_dd_limit_pct):")
    print(grid_sharpe.to_string())
    print("\nMax DD %:")
    print(grid_dd.to_string())
    print(f"\nSaved to {OUT_DIR}\\streak_cost_grid.csv")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeline', default=TIMELINE)
    a = ap.parse_args()
    TIMELINE = a.timeline
    main()
