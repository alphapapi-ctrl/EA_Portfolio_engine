"""
test_relative_rules.py
======================
Relative (per-EA baseline) benching rules on the engine side, both honest forms:

1. Expanding: full-period sim; each EA's baseline is its worst COMPLETED
   streak so far in the sim (no look-ahead, judged only against its own past).
2. Walk-forward: baselines frozen from the 2020-22 calibration window, rules
   run frozen on 2023-26 — the exact analog of live usage, where baselines
   come from backtests that precede the live period. Row appended to the
   walk-forward comparison table.

Both use relative-only benching (no absolute streak/DD rules) + corr cap 0.7,
so the relative rules' contribution is isolated.
"""

import os

import pandas as pd

from portfolio_sim import (load_timeline, simulate, ea_stats, pick_top,
                           day_streak_baseline, REGIMES)
from run_sim import run_one

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
WF_DIR     = os.path.join(ENGINE_DIR, 'runs', '_walk_forward')

N_SLOTS, GROSS, REVIEW, WARMUP = 10, 10.0, 5, 63
RATIO = 1.0


def main():
    daily, _ = load_timeline('main_pool')
    candidates = list(daily.columns)

    # ── 1. Expanding baselines, full period ───────────────────────────────
    portfolio = pick_top(ea_stats(daily.iloc[:WARMUP]), 'sharpe', N_SLOTS)
    regime = REGIMES['rules'](
        list(portfolio), candidates, GROSS, lookback=63, metric='sharpe',
        loss_streak_limit=None, streak_dollar_limit=None, ea_dd_limit_pct=None,
        corr_cap=0.7, cooldown_days=21,
        relative_ratio=RATIO, rel_expanding=True)
    res = simulate(daily, regime, review_every=REVIEW, warmup=WARMUP)
    s = res['summary']
    print('1) EXPANDING baselines, full period:')
    print('  ', {k: s[k] for k in ['sharpe', 'ann_return_pct', 'max_dd_pct',
                                   'turnover_units', 'events']})
    drops = [e for e in res['events'] if e['action'] == 'drop']
    print(f'   drops: {len(drops)} | sample reasons:')
    for e in drops[:4]:
        print('    -', e['detail'])

    # persist as a normal run for the Results page
    out = os.path.join(ENGINE_DIR, 'runs', 'rules_relative_expanding')
    os.makedirs(out, exist_ok=True)
    res['equity'].to_csv(os.path.join(out, 'equity.csv'))
    res['weights'].to_csv(os.path.join(out, 'weights.csv'))
    pd.DataFrame(res['events']).to_csv(os.path.join(out, 'events.csv'), index=False)
    import json
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump({'config': {'regime': 'rules', 'relative_ratio': RATIO,
                              'rel_expanding': True, 'corr_cap': 0.7},
                   'summary': s}, f, indent=2)

    # ── 2. Walk-forward: baselines frozen on calibration window ──────────
    split = int((daily.index < '2023-01-01').sum())
    calib = daily.iloc[:split]
    baselines = {ea: day_streak_baseline(calib[ea]) for ea in daily.columns}
    portfolio = pick_top(ea_stats(calib), 'sharpe', N_SLOTS)

    regime = REGIMES['rules'](
        list(portfolio), candidates, GROSS, lookback=63, metric='sharpe',
        loss_streak_limit=None, streak_dollar_limit=None, ea_dd_limit_pct=None,
        corr_cap=0.7, cooldown_days=21,
        relative_ratio=RATIO, rel_baselines=baselines)
    res = simulate(daily, regime, review_every=REVIEW, warmup=split)
    s = res['summary']
    print('\n2) WALK-FORWARD, baselines frozen on 2020-22, test 2023-26:')
    print('  ', {k: s[k] for k in ['sharpe', 'ann_return_pct', 'max_dd_pct',
                                   'turnover_units', 'events']})
    drops = [e for e in res['events'] if e['action'] == 'drop']
    print(f'   drops: {len(drops)} | sample reasons:')
    for e in drops[:4]:
        print('    -', e['detail'])

    row = {'run': f'rules relative baselines (ratio {RATIO})',
           **{k: s[k] for k in ['net_profit', 'ann_return_pct', 'ann_vol_pct',
                                'sharpe', 'max_dd', 'max_dd_pct', 'calmar',
                                'turnover_units', 'events']}}
    cmp_path = os.path.join(WF_DIR, 'test_window_comparison.csv')
    table = pd.read_csv(cmp_path)
    table = table[table['run'] != row['run']]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table = table.sort_values('sharpe', ascending=False)
    table.to_csv(cmp_path, index=False)
    print('\nUpdated walk-forward table:')
    print(table[['run', 'sharpe', 'ann_return_pct', 'max_dd_pct',
                 'turnover_units']].to_string(index=False))


if __name__ == '__main__':
    main()
