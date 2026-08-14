"""
monte_carlo.py
==============
EA Portfolio Engine — block-bootstrap Monte Carlo.

Resamples whole ROWS of the daily P&L matrix (keeping every EA's P&L on a
given day together, so cross-EA correlation is preserved) in contiguous
blocks, rebuilds synthetic histories of the same length, and runs the frozen
rules regime + equal-weight benchmark through each.

Block sizes test how much each regime depends on sequence persistence:
  5d  (weekly)    — scrambles regimes hard
  21d (monthly)   — keeps short regimes
  63d (quarterly) — keeps regime chunks intact

Outputs percentile tables to runs/_monte_carlo/.
"""

import os

import numpy as np
import pandas as pd

from portfolio_sim import load_timeline, simulate, ea_stats, rank_metric, REGIMES

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(ENGINE_DIR, 'runs', '_monte_carlo')

TIMELINE     = 'main_pool'
BLOCK_SIZES  = {'weekly_5d': 5, 'monthly_21d': 21, 'quarterly_63d': 63}
N_ITER       = 100
N_SLOTS      = 10
GROSS        = 10.0
REVIEW_EVERY = 5
WARMUP       = 63
RULES_PARAMS = dict(lookback=63, metric='sharpe', loss_streak_limit=5,
                    ea_dd_limit_pct=2.5, corr_cap=0.7, cooldown_days=21)


def block_resample(daily, block, rng):
    """Synthetic history: contiguous row-blocks sampled with replacement."""
    n = len(daily)
    starts = rng.integers(0, n - block, size=(n // block) + 1)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    out = daily.iloc[idx].copy()
    out.index = daily.index  # keep the original calendar for the simulator
    return out


def run_sample(sample):
    """Run frozen rules + equal weight on one synthetic history."""
    candidates = list(sample.columns)
    stats      = ea_stats(sample.iloc[:WARMUP])
    portfolio  = rank_metric(stats, 'sharpe').nlargest(N_SLOTS).index.tolist()

    rules = REGIMES['rules'](portfolio, candidates, GROSS, **RULES_PARAMS)
    s_rules = simulate(sample, rules, review_every=REVIEW_EVERY, warmup=WARMUP)['summary']

    ew = REGIMES['equal_weight'](candidates, GROSS)
    s_ew = simulate(sample, ew, review_every=REVIEW_EVERY, warmup=WARMUP)['summary']
    return s_rules, s_ew


def pct_table(df):
    out = {}
    for col in ['net_profit', 'sharpe', 'max_dd_pct']:
        out[col] = {
            'p05'   : round(float(df[col].quantile(0.05)), 2),
            'median': round(float(df[col].median()), 2),
            'p95'   : round(float(df[col].quantile(0.95)), 2),
        }
    out['prob_dd_gt_5pct']  = round(float((df['max_dd_pct'] > 5).mean()), 3)
    out['prob_dd_gt_10pct'] = round(float((df['max_dd_pct'] > 10).mean()), 3)
    out['prob_loss']        = round(float((df['net_profit'] < 0).mean()), 3)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    daily, _ = load_timeline(TIMELINE)
    print(f"Timeline {TIMELINE}: {daily.shape[0]} days x {daily.shape[1]} EAs")
    print(f"{N_ITER} iterations per block size\n")

    summary_rows = []
    for label, block in BLOCK_SIZES.items():
        print(f"[{label}] block={block}d ...", flush=True)
        rng = np.random.default_rng(7)
        rules_rows, ew_rows = [], []
        for i in range(N_ITER):
            sample = block_resample(daily, block, rng)
            s_rules, s_ew = run_sample(sample)
            rules_rows.append(s_rules)
            ew_rows.append(s_ew)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{N_ITER}", flush=True)

        rules_df = pd.DataFrame(rules_rows)
        ew_df    = pd.DataFrame(ew_rows)
        rules_df.to_csv(os.path.join(OUT_DIR, f'rules_{label}.csv'), index=False)
        ew_df.to_csv(os.path.join(OUT_DIR, f'equal_weight_{label}.csv'), index=False)

        for regime, df in [('rules', rules_df), ('equal_weight', ew_df)]:
            t = pct_table(df)
            summary_rows.append({
                'block': label, 'regime': regime,
                'sharpe_p05': t['sharpe']['p05'],
                'sharpe_med': t['sharpe']['median'],
                'sharpe_p95': t['sharpe']['p95'],
                'dd_pct_p05': t['max_dd_pct']['p05'],
                'dd_pct_med': t['max_dd_pct']['median'],
                'dd_pct_p95': t['max_dd_pct']['p95'],
                'net_med'   : t['net_profit']['median'],
                'P(dd>5%)'  : t['prob_dd_gt_5pct'],
                'P(dd>10%)' : t['prob_dd_gt_10pct'],
                'P(loss)'   : t['prob_loss'],
            })

    summary = pd.DataFrame(summary_rows).set_index(['block', 'regime'])
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'))
    print("\n=== MONTE CARLO SUMMARY ===")
    print(summary.to_string())
    print(f"\nSaved to {OUT_DIR}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeline', default=TIMELINE)
    a = ap.parse_args()
    TIMELINE = a.timeline
    main()
