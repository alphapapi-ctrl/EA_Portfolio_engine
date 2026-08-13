"""
build_regime_matrix.py
======================
Condition each EA's daily P&L on market regime states built from the
reference instruments (fetch_reference_data.py must have run first).

Regime states (simple, transparent, no look-ahead beyond the moving average):
  DXY    : strong dollar / weak dollar          (price vs 100-day average)
  VIX    : calm (<15) / normal (15-25) / stressed (>25)
  SP500  : stocks bull / stocks bear            (price vs 200-day average)
  GOLD   : gold uptrend / gold downtrend        (price vs 200-day average)
  BTC    : crypto bull / crypto bear            (price vs 200-day average)
  US10Y  : rates rising / rates falling         (yield vs 100-day average)
  OIL    : oil uptrend / oil downtrend          (price vs 200-day average)

Outputs (per timeline):
  timeline/<name>/regime_states.csv   date x indicator state calendar
  timeline/<name>/regime_matrix.csv   long-form: row per EA (plus per family
                                      and the whole pool) x indicator x state
                                      with days, total_pnl, pnl_share, sharpe

This is DESCRIPTIVE: it reports where each robot's profit historically
happened. It says nothing about prediction — regime states are coincident.

Usage: python build_regime_matrix.py [--timeline main_pool]
"""

import os
import argparse

import numpy as np
import pandas as pd

from portfolio_sim import load_timeline, TRADING_DAYS

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_FILE   = os.path.join(ENGINE_DIR, 'reference', 'reference_prices.csv')

STATE_DEFS = {
    'DXY'  : ('sma', 100, 'strong dollar', 'weak dollar'),
    'SP500': ('sma', 200, 'stocks bull', 'stocks bear'),
    'GOLD' : ('sma', 200, 'gold uptrend', 'gold downtrend'),
    'BTC'  : ('sma', 200, 'crypto bull', 'crypto bear'),
    'US10Y': ('sma', 100, 'rates rising', 'rates falling'),
    'OIL'  : ('sma', 200, 'oil uptrend', 'oil downtrend'),
    'VIX'  : ('levels', (15, 25), 'calm', 'normal', 'stressed'),
}


def build_states(prices):
    states = {}
    for ind, spec in STATE_DEFS.items():
        if ind not in prices.columns:
            continue
        s = prices[ind].dropna()
        if spec[0] == 'sma':
            _, window, above, below = spec
            sma = s.rolling(window).mean()
            states[ind] = pd.Series(np.where(s >= sma, above, below),
                                    index=s.index).where(sma.notna())
        else:
            _, (lo, hi), low_lbl, mid_lbl, high_lbl = spec
            states[ind] = pd.Series(
                np.select([s < lo, s > hi], [low_lbl, high_lbl], mid_lbl),
                index=s.index)
    return pd.DataFrame(states)


def condition(pnl, state_series):
    """Stats of a daily P&L series per state. pnl and states share an index."""
    rows = []
    for state, seg in pnl.groupby(state_series):
        vol = seg.std(ddof=0)
        rows.append({
            'state'    : state,
            'days'     : int(len(seg)),
            'total_pnl': round(float(seg.sum()), 2),
            'avg_daily': round(float(seg.mean()), 2),
            'sharpe'   : round(float(seg.mean() / vol * np.sqrt(TRADING_DAYS)), 2)
                         if vol > 0 else 0.0,
        })
    total = sum(abs(r['total_pnl']) for r in rows) or 1.0
    for r in rows:
        r['pnl_share_pct'] = round(r['total_pnl'] / sum(x['total_pnl'] for x in rows) * 100, 1) \
                             if sum(x['total_pnl'] for x in rows) != 0 else 0.0
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeline', default='main_pool')
    args = ap.parse_args()

    daily, meta = load_timeline(args.timeline)
    prices = pd.read_csv(REF_FILE, index_col='date', parse_dates=['date'])
    states = build_states(prices)
    # Align to the P&L calendar (forward-fill weekends/holidays)
    states = states.reindex(states.index.union(daily.index)).ffill().reindex(daily.index)

    tdir = os.path.join(ENGINE_DIR, 'timeline', args.timeline)
    states.to_csv(os.path.join(tdir, 'regime_states.csv'), index_label='date')

    active_range = {r.ea_id: (pd.Timestamp(r.first_trade), pd.Timestamp(r.last_trade))
                    for r in meta.itertuples()}

    records = []

    def add_entity(entity_id, entity_type, pnl):
        for ind in states.columns:
            for row in condition(pnl, states[ind]):
                records.append({'entity': entity_id, 'type': entity_type,
                                'indicator': ind, **row})

    # Per EA, restricted to its active range
    for ea in daily.columns:
        lo, hi = active_range.get(ea, (daily.index[0], daily.index[-1]))
        add_entity(ea, 'ea', daily[ea].loc[lo:hi])

    # Per family (equal-weight sum of members) and whole pool
    for fam, grp in meta.groupby('family'):
        cols = [c for c in grp.ea_id if c in daily.columns]
        if cols:
            add_entity(f'FAMILY: {fam}', 'family', daily[cols].sum(axis=1))
    add_entity('POOL: all robots', 'pool', daily.sum(axis=1))

    out = pd.DataFrame(records)
    out.to_csv(os.path.join(tdir, 'regime_matrix.csv'), index=False)

    print(f"Timeline {args.timeline}: {len(daily.columns)} EAs, "
          f"{len(states.columns)} indicators")
    print(f"State days (of {len(states)}):")
    for ind in states.columns:
        print(f"  {ind:6s} " + '  '.join(f'{k}: {v}' for k, v in
              states[ind].value_counts().items()))
    print(f"\nSaved regime_states.csv + regime_matrix.csv "
          f"({len(out)} rows) -> {tdir}")


if __name__ == '__main__':
    main()
