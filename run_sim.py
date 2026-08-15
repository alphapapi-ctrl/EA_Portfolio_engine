"""
run_sim.py
==========
EA Portfolio Engine — run portfolio simulations from configs.

A run config (JSON) picks a timeline, a regime + params, review cadence and
optional overlays. Results land in runs/<run_name>/: equity.csv, weights.csv,
events.csv, summary.json.

Usage:
    python run_sim.py --preset all --timeline main_pool     # benchmark suite
    python run_sim.py --config runs/my_run.json             # single custom run
    python run_sim.py --compare                             # table of all runs

Config example (runs/my_run.json):
{
  "timeline"     : "main_pool",
  "regime"       : "rules",
  "portfolio"    : ["<ea_id>", ...],          // or "top10_sharpe" / "all"
  "substitutes"  : "all",                     // pool to swap in from
  "n_slots"      : 10,
  "gross_budget" : 10.0,                      // total risk units deployed
  "review_every" : 5,
  "warmup"       : 63,
  "params"       : {"lookback": 63, "metric": "sharpe",
                    "loss_streak_limit": 5, "ea_dd_limit_pct": 2.5,
                    "corr_cap": 0.7, "cooldown_days": 21},
  "overlays"     : {"vol_target": {"target_ann_vol": 15000},
                    "dd_derisk": {"start_pct": 3.0, "floor_pct": 6.0}}
}
"""

import os
import sys
import json
import argparse

import pandas as pd

from portfolio_sim import (load_timeline, load_trades, simulate, ea_stats,
                           rank_metric, pick_top, REGIMES, OVERLAYS)

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR   = os.path.join(ENGINE_DIR, 'runs')


# ── Config resolution ─────────────────────────────────────────────────────────

def resolve_pool(spec, daily, warmup, max_per_symbol=None):
    """Resolve a portfolio/substitutes spec into a list of ea_ids."""
    if spec == 'all' or spec is None:
        return list(daily.columns)
    if isinstance(spec, str) and spec.startswith('top'):
        # e.g. "top10_sharpe" — ranked over the warmup window only (no look-ahead)
        n, metric = spec[3:].split('_')
        stats = ea_stats(daily.iloc[:warmup])
        return pick_top(stats, metric, int(n), max_per_symbol)
    return list(spec)


def build_regime(cfg, daily):
    name    = cfg['regime']
    gross   = float(cfg.get('gross_budget', cfg.get('n_slots', 10)))
    n_slots = int(cfg.get('n_slots', 10))
    warmup  = int(cfg.get('warmup', 63))
    params  = dict(cfg.get('params', {}))

    max_sym    = params.get('max_per_symbol')
    candidates = resolve_pool(cfg.get('portfolio', 'all'), daily, warmup,
                              max_per_symbol=max_sym)

    if name == 'equal_weight':
        return EqualWeightSafe(candidates, gross)
    if name == 'random':
        return REGIMES['random'](candidates, gross, n_slots,
                                 seed=int(cfg.get('seed', 42)))
    if name == 'static_topn':
        return REGIMES['static_topn'](candidates, gross, n_slots,
                                      metric=params.get('metric', 'sharpe'))
    if name == 'momentum':
        return REGIMES['momentum'](candidates, gross, n_slots,
                                   lookback=int(params.get('lookback', 63)),
                                   metric=params.get('metric', 'sharpe'),
                                   max_per_symbol=max_sym,
                                   streak_dollar_limit=params.get('streak_dollar_limit'))
    if name == 'inverse_vol':
        return REGIMES['inverse_vol'](candidates, gross,
                                      lookback=int(params.get('lookback', 63)))
    if name == 'rules':
        subs = resolve_pool(cfg.get('substitutes', 'all'), daily, warmup)
        subs = [s for s in subs if s not in candidates] + candidates
        tradebook = None
        if params.get('streak_mode') == 'trades':
            tradebook = load_trades(cfg['timeline'])
        return REGIMES['rules'](
            candidates[:n_slots], subs, gross,
            lookback            = int(params.get('lookback', 63)),
            metric              = params.get('metric', 'sharpe'),
            loss_streak_limit   = params.get('loss_streak_limit'),
            ea_dd_limit_pct     = params.get('ea_dd_limit_pct'),
            corr_cap            = params.get('corr_cap'),
            cooldown_days       = int(params.get('cooldown_days', 21)),
            max_per_symbol      = max_sym,
            streak_mode         = params.get('streak_mode', 'days'),
            streak_dollar_limit = params.get('streak_dollar_limit'),
            loss_count_limit    = params.get('loss_count_limit'),
            loss_count_window   = int(params.get('loss_count_window', 21)),
            tradebook           = tradebook,
            capacity            = int(cfg.get('capacity', 0)) or None,
            fill_blanks_after   = int(cfg.get('fill_blanks_after', 0)),
        )
    raise ValueError(f"Unknown regime: {name}")


def EqualWeightSafe(candidates, gross):
    return REGIMES['equal_weight'](candidates, gross)


def build_overlays(cfg):
    out = []
    for name, params in (cfg.get('overlays') or {}).items():
        if params:
            out.append(OVERLAYS[name](**params))
    return out


# ── Run + persist ─────────────────────────────────────────────────────────────

def run_one(run_name, cfg, daily):
    # Optional test-period slice — everything (including warmup and any
    # top-N auto-pick) sees only data inside the window.
    sd, ed = cfg.get('start_date'), cfg.get('end_date')
    if sd or ed:
        daily = daily.loc[sd or daily.index[0]: ed or daily.index[-1]]
        if len(daily) < int(cfg.get('warmup', 63)) + 10:
            raise ValueError('Test period too short for the warm-up window.')
    regime   = build_regime(cfg, daily)
    overlays = build_overlays(cfg)
    result   = simulate(daily, regime,
                        review_every = int(cfg.get('review_every', 5)),
                        warmup       = int(cfg.get('warmup', 63)),
                        overlays     = overlays)

    out = os.path.join(RUNS_DIR, run_name)
    os.makedirs(out, exist_ok=True)
    result['equity'].to_csv(os.path.join(out, 'equity.csv'))
    result['weights'].to_csv(os.path.join(out, 'weights.csv'))
    pd.DataFrame(result['events']).to_csv(os.path.join(out, 'events.csv'), index=False)
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump({'config': cfg, 'summary': result['summary']}, f, indent=2, default=str)
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    return result['summary']


# ── Preset suite ──────────────────────────────────────────────────────────────

def preset_configs(timeline):
    base = {'timeline': timeline, 'n_slots': 10, 'gross_budget': 10.0,
            'review_every': 5, 'warmup': 63}
    return {
        'bench_equal_weight': {**base, 'regime': 'equal_weight'},
        'bench_random'      : {**base, 'regime': 'random', 'seed': 42},
        'bench_static_top10': {**base, 'regime': 'static_topn',
                               'params': {'metric': 'sharpe'}},
        'momentum_3m'       : {**base, 'regime': 'momentum',
                               'params': {'lookback': 63, 'metric': 'sharpe'}},
        'momentum_6m'       : {**base, 'regime': 'momentum',
                               'params': {'lookback': 126, 'metric': 'sharpe'}},
        'inverse_vol'       : {**base, 'regime': 'inverse_vol',
                               'params': {'lookback': 63}},
        'rules_default'     : {**base, 'regime': 'rules',
                               'portfolio': 'top10_sharpe', 'substitutes': 'all',
                               'params': {'lookback': 63, 'metric': 'sharpe',
                                          'loss_streak_limit': 5,
                                          'ea_dd_limit_pct': 2.5,
                                          'corr_cap': None, 'cooldown_days': 21}},
        'rules_corr_cap'    : {**base, 'regime': 'rules',
                               'portfolio': 'top10_sharpe', 'substitutes': 'all',
                               'params': {'lookback': 63, 'metric': 'sharpe',
                                          'loss_streak_limit': 5,
                                          'ea_dd_limit_pct': 2.5,
                                          'corr_cap': 0.7, 'cooldown_days': 21}},
        'momentum_3m_ddctl' : {**base, 'regime': 'momentum',
                               'params': {'lookback': 63, 'metric': 'sharpe'},
                               'overlays': {'dd_derisk': {'start_pct': 3.0,
                                                          'floor_pct': 6.0}}},
    }


def print_comparison(summaries):
    cols = ['net_profit', 'ann_return_pct', 'ann_vol_pct', 'sharpe',
            'max_dd', 'max_dd_pct', 'calmar', 'turnover_units', 'events']
    df = pd.DataFrame(summaries).T[cols]
    df = df.sort_values('sharpe', ascending=False)
    print()
    print(df.to_string())


def main():
    ap = argparse.ArgumentParser(description='Run EA portfolio simulations.')
    ap.add_argument('--config',   help='Path to a single run config JSON')
    ap.add_argument('--preset',   help='"all" to run the benchmark/regime suite')
    ap.add_argument('--timeline', default='main_pool')
    ap.add_argument('--compare',  action='store_true',
                    help='Print comparison table of all completed runs')
    args = ap.parse_args()

    if args.compare:
        summaries = {}
        for d in sorted(os.listdir(RUNS_DIR)):
            p = os.path.join(RUNS_DIR, d, 'summary.json')
            if os.path.isfile(p):
                with open(p) as f:
                    summaries[d] = json.load(f)['summary']
        if summaries:
            print_comparison(summaries)
        else:
            print('  No completed runs found.')
        return

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        run_name = os.path.splitext(os.path.basename(args.config))[0]
        daily, _ = load_timeline(cfg.get('timeline', args.timeline))
        summary = run_one(run_name, cfg, daily)
        print(json.dumps(summary, indent=2))
        return

    if args.preset == 'all':
        daily, _ = load_timeline(args.timeline)
        print(f"  Timeline {args.timeline}: {daily.shape[0]} days x {daily.shape[1]} EAs")
        summaries = {}
        for name, cfg in preset_configs(args.timeline).items():
            print(f"  Running {name} ...")
            summaries[name] = run_one(name, cfg, daily)
        print_comparison(summaries)
        return

    ap.print_help()


if __name__ == '__main__':
    main()
