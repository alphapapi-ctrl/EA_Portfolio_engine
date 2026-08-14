"""
bench_realistic_ew.py
=====================
The realistic passive benchmark: nobody runs all 140 robots — a real account
holds a BUCKET of 10-25. This measures equal-weight-and-hold over random
subsets of realistic size (10 seeds per size), at the same total risk budget
as the managed regimes (gross 10 units spread across the bucket), on:

  - the full window (warmup 63), and
  - the out-of-sample window only (2023+, matching the walk-forward table)

so the "doing nothing" bar can be quoted at attainable diversification levels
instead of the unattainable 140-robot version.

Saves runs/_sweeps/realistic_ew.csv.
"""

import os

import numpy as np
import pandas as pd

from portfolio_sim import load_timeline, simulate, REGIMES

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(ENGINE_DIR, 'runs', '_sweeps')

TIMELINE   = 'main_pool_2018'
TEST_START = '2023-01-01'
SIZES      = [10, 15, 20, 25]
SEEDS      = range(1, 11)
GROSS      = 10.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    daily, _ = load_timeline(TIMELINE)
    split = int((daily.index < TEST_START).sum())
    cols = list(daily.columns)

    rows = []
    for size in SIZES:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            bucket = list(rng.choice(cols, size=size, replace=False))
            for window, warmup in [('full', 63), ('oos_2023', split)]:
                regime = REGIMES['equal_weight'](bucket, GROSS)
                s = simulate(daily, regime, review_every=5,
                             warmup=warmup)['summary']
                rows.append({'size': size, 'seed': seed, 'window': window, **s})
        print(f'  size {size} done', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'realistic_ew.csv'), index=False)

    for window in ['full', 'oos_2023']:
        sub = df[df.window == window]
        agg = sub.groupby('size').agg(
            sharpe_mean=('sharpe', 'mean'), sharpe_min=('sharpe', 'min'),
            sharpe_max=('sharpe', 'max'),
            dd_mean=('max_dd_pct', 'mean'), dd_worst=('max_dd_pct', 'max'),
            ret_mean=('ann_return_pct', 'mean'))
        print(f"\n=== equal-weight-and-hold random buckets — {window} ===")
        print(agg.round(2).to_string())


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeline', default=TIMELINE)
    a = ap.parse_args()
    TIMELINE = a.timeline
    main()
