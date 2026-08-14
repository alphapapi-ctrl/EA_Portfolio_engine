"""
sweep_review_cadence.py
=======================
Sweep the review cadence (review_every, in trading days) — the only knob the
other sweeps held fixed (at 5). Runs three regimes at each cadence:
  - rules, multi-rule   (streak 5 days + DD 2.5%, corr cap 0.7)
  - rules, cost-only    (streak cost $1,000, corr cap 0.7)
  - momentum            (lookback 63)

Saves runs/_sweeps/review_cadence.csv and prints the table.
"""

import os

import pandas as pd

from portfolio_sim import load_timeline, simulate, ea_stats, pick_top, REGIMES

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(ENGINE_DIR, 'runs', '_sweeps')

TIMELINE = 'main_pool'
N_SLOTS  = 10
GROSS    = 10.0
WARMUP   = 63
CADENCES = [1, 2, 3, 5, 10, 15, 21]


def regimes(portfolio, candidates):
    return {
        'rules_multi': lambda: REGIMES['rules'](
            list(portfolio), candidates, GROSS, lookback=63, metric='sharpe',
            loss_streak_limit=5, ea_dd_limit_pct=2.5,
            corr_cap=0.7, cooldown_days=21),
        'rules_cost' : lambda: REGIMES['rules'](
            list(portfolio), candidates, GROSS, lookback=63, metric='sharpe',
            loss_streak_limit=None, ea_dd_limit_pct=None,
            streak_dollar_limit=1000, corr_cap=0.7, cooldown_days=21),
        'momentum'   : lambda: REGIMES['momentum'](
            candidates, GROSS, N_SLOTS, lookback=63, metric='sharpe'),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    daily, _ = load_timeline(TIMELINE)
    candidates = list(daily.columns)
    portfolio  = pick_top(ea_stats(daily.iloc[:WARMUP]), 'sharpe', N_SLOTS)

    rows = []
    for cadence in CADENCES:
        for name, make in regimes(portfolio, candidates).items():
            s = simulate(daily, make(), review_every=cadence,
                         warmup=WARMUP)['summary']
            rows.append({'review_every': cadence, 'regime': name, **s})
        print(f"  cadence {cadence} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'review_cadence.csv'), index=False)
    for name in df.regime.unique():
        sub = df[df.regime == name].set_index('review_every')
        print(f"\n{name}:")
        print(sub[['sharpe', 'ann_return_pct', 'max_dd_pct',
                   'turnover_units', 'events']].to_string())
    print(f"\nSaved to {OUT_DIR}\\review_cadence.csv")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeline', default=TIMELINE)
    a = ap.parse_args()
    TIMELINE = a.timeline
    main()
