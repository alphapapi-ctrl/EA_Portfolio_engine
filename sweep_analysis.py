"""
sweep_analysis.py
=================
EA Portfolio Engine — robustness checks on the preset results:

1. Random-rotation control averaged over many seeds (is momentum's edge
   bigger than seed luck?).
2. Rules-regime parameter grid: loss_streak_limit x ea_dd_limit_pct — is the
   good result a stable region or a lucky spike?
3. Momentum lookback sweep.
4. Correlation-cap sweep on the rules regime.

Saves tables to runs/_sweeps/ and prints them.
"""

import os
import json

import numpy as np
import pandas as pd

from portfolio_sim import load_timeline, simulate, ea_stats, rank_metric, REGIMES

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(ENGINE_DIR, 'runs', '_sweeps')

TIMELINE     = 'main_pool'
N_SLOTS      = 10
GROSS        = 10.0
REVIEW_EVERY = 5
WARMUP       = 63


def top10_sharpe(daily):
    stats = ea_stats(daily.iloc[:WARMUP])
    return rank_metric(stats, 'sharpe').nlargest(N_SLOTS).index.tolist()


def run(daily, regime):
    return simulate(daily, regime, review_every=REVIEW_EVERY, warmup=WARMUP)['summary']


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    daily, _ = load_timeline(TIMELINE)
    print(f"Timeline {TIMELINE}: {daily.shape[0]} days x {daily.shape[1]} EAs")
    candidates = list(daily.columns)
    portfolio  = top10_sharpe(daily)

    # ── 1. Random control, many seeds ────────────────────────────────────
    print("\n[1/4] Random rotation x 10 seeds ...")
    rows = []
    for seed in range(1, 11):
        s = run(daily, REGIMES['random'](candidates, GROSS, N_SLOTS, seed=seed))
        rows.append({'seed': seed, **s})
    rand = pd.DataFrame(rows).set_index('seed')
    rand.to_csv(os.path.join(OUT_DIR, 'random_seeds.csv'))
    print(rand[['sharpe', 'ann_return_pct', 'max_dd_pct']].to_string())
    print(f"  sharpe mean {rand.sharpe.mean():.2f}  std {rand.sharpe.std():.2f}  "
          f"min {rand.sharpe.min():.2f}  max {rand.sharpe.max():.2f}")

    # ── 2. Rules grid: loss streak x DD limit (corr cap 0.7) ─────────────
    print("\n[2/4] Rules grid: loss_streak x ea_dd_limit (corr_cap=0.7) ...")
    streaks   = [3, 4, 5, 6, 7, None]
    dd_limits = [1.5, 2.0, 2.5, 3.0, 4.0, None]
    grid_sharpe = pd.DataFrame(index=[str(s) for s in streaks],
                               columns=[str(d) for d in dd_limits], dtype=float)
    grid_dd = grid_sharpe.copy()
    records = []
    for st in streaks:
        for dd in dd_limits:
            if st is None and dd is None:
                continue  # no drop rule at all = static portfolio
            regime = REGIMES['rules'](
                list(portfolio), candidates, GROSS,
                lookback=63, metric='sharpe',
                loss_streak_limit=st, ea_dd_limit_pct=dd,
                corr_cap=0.7, cooldown_days=21)
            s = run(daily, regime)
            grid_sharpe.loc[str(st), str(dd)] = s['sharpe']
            grid_dd.loc[str(st), str(dd)]     = s['max_dd_pct']
            records.append({'loss_streak': st, 'ea_dd_limit_pct': dd, **s})
    pd.DataFrame(records).to_csv(os.path.join(OUT_DIR, 'rules_grid.csv'), index=False)
    print("  Sharpe (rows=loss_streak, cols=ea_dd_limit_pct):")
    print(grid_sharpe.to_string())
    print("  Max DD % :")
    print(grid_dd.to_string())

    # ── 3. Momentum lookback sweep ───────────────────────────────────────
    print("\n[3/4] Momentum lookback sweep ...")
    rows = []
    for lb in [21, 42, 63, 126, 189, 252]:
        s = run(daily, REGIMES['momentum'](candidates, GROSS, N_SLOTS,
                                           lookback=lb, metric='sharpe'))
        rows.append({'lookback': lb, **s})
    mom = pd.DataFrame(rows).set_index('lookback')
    mom.to_csv(os.path.join(OUT_DIR, 'momentum_lookback.csv'))
    print(mom[['sharpe', 'ann_return_pct', 'max_dd_pct', 'turnover_units']].to_string())

    # ── 4. Correlation cap sweep (rules, streak=5, dd=2.5) ───────────────
    print("\n[4/4] Correlation cap sweep ...")
    rows = []
    for cap in [None, 0.5, 0.6, 0.7, 0.8, 0.9]:
        regime = REGIMES['rules'](
            list(portfolio), candidates, GROSS,
            lookback=63, metric='sharpe',
            loss_streak_limit=5, ea_dd_limit_pct=2.5,
            corr_cap=cap, cooldown_days=21)
        s = run(daily, regime)
        rows.append({'corr_cap': str(cap), **s})
    cc = pd.DataFrame(rows).set_index('corr_cap')
    cc.to_csv(os.path.join(OUT_DIR, 'corr_cap.csv'))
    print(cc[['sharpe', 'ann_return_pct', 'max_dd_pct', 'events']].to_string())

    print(f"\nSaved tables to {OUT_DIR}")


if __name__ == '__main__':
    main()
