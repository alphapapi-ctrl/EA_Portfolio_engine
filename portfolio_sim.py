"""
portfolio_sim.py
================
EA Portfolio Engine — simulator core.

Walks the daily P&L matrix day by day. A regime is asked for a weight vector
at each review date using ONLY history up to the previous day (no look-ahead);
portfolio P&L that day = sum(weight_i x pnl_i). Weights are risk units:
1.0 = the EA exactly as backtested (lot step calibrated to the 5% DD target),
and P&L scales linearly because every backtest ran fixed-balance, fixed-lot.

Equity is tracked against a fixed basis (no compounding) so results stay
linear and comparable across regimes.
"""

import os
import re
import json

import numpy as np
import pandas as pd

ENGINE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRADING_DAYS = 252

# ea_ids end "_<SYMBOL>.a_<period>_<model>" — extract the market
_SYMBOL_RE = re.compile(r'_([A-Za-z0-9.]+\.a)_[A-Za-z0-9]+_[A-Z]+$')


def symbol_of(ea_id):
    m = _SYMBOL_RE.search(ea_id)
    return m.group(1) if m else ea_id


def pick_top(stats, metric, n, max_per_symbol=None):
    """Top-n EAs by metric, optionally capping how many share one market."""
    ranked = rank_metric(stats, metric).sort_values(ascending=False)
    team = []
    for ea in ranked.index:
        if len(team) >= n:
            break
        if max_per_symbol:
            sym = symbol_of(ea)
            if sum(1 for t in team if symbol_of(t) == sym) >= max_per_symbol:
                continue
        team.append(ea)
    return team


# ── Timeline loading ──────────────────────────────────────────────────────────

def load_timeline(name):
    tdir  = os.path.join(ENGINE_DIR, 'timeline', name)
    daily = pd.read_csv(os.path.join(tdir, 'daily_pnl.csv'),
                        index_col='date', parse_dates=['date'])
    meta  = pd.read_csv(os.path.join(tdir, 'ea_meta.csv'))
    return daily, meta


def load_trades(name):
    """Per-trade history for trade-based streak rules."""
    tdir = os.path.join(ENGINE_DIR, 'timeline', name)
    t = pd.read_csv(os.path.join(tdir, 'trades.csv'),
                    usecols=['ea_id', 'close_time', 'net_profit'],
                    parse_dates=['close_time'])
    return TradeBook(t)


class TradeBook:
    """Fast per-EA trade lookups for streak rules counted in trades."""

    def __init__(self, trades_df):
        self.by_ea = {}
        for ea, g in trades_df.sort_values('close_time').groupby('ea_id'):
            self.by_ea[ea] = (g['close_time'].to_numpy(), g['net_profit'].to_numpy())

    def streak(self, ea, asof):
        """(length, $ total) of the consecutive losing-trade run ending at
        the last trade closed on or before `asof`."""
        if ea not in self.by_ea:
            return 0, 0.0
        times, pnl = self.by_ea[ea]
        i = int(np.searchsorted(times, np.datetime64(asof), side='right')) - 1
        n, total = 0, 0.0
        while i >= 0 and pnl[i] < 0:
            n += 1
            total += float(pnl[i])
            i -= 1
        return n, total

    def losses_between(self, ea, start, end):
        """Number of losing trades closed in (start, end]."""
        if ea not in self.by_ea:
            return 0
        times, pnl = self.by_ea[ea]
        lo = int(np.searchsorted(times, np.datetime64(start), side='left'))
        hi = int(np.searchsorted(times, np.datetime64(end), side='right'))
        return int((pnl[lo:hi] < 0).sum())


def day_streak_stats(col):
    """(length, $ total) of the trailing run of consecutive losing days,
    ignoring days with no trades."""
    vals = col.to_numpy()
    vals = vals[vals != 0.0]
    n, total = 0, 0.0
    for v in vals[::-1]:
        if v < 0:
            n += 1
            total += float(v)
        else:
            break
    return n, total


def day_streak_baseline(col):
    """
    Worst COMPLETED losing-day streak (length, $ cost) in a series, excluding
    the trailing (possibly still-running) streak — so a live streak is judged
    against records set before it, never against itself.
    """
    vals = col.to_numpy()
    vals = vals[vals != 0.0]
    runs, n, c = [], 0, 0.0
    for v in vals:
        if v < 0:
            n += 1
            c += float(v)
        else:
            if n:
                runs.append((n, c))
            n, c = 0, 0.0
    if not runs:                    # only the ongoing run (or none) exists
        return 0, 0.0
    return max(r[0] for r in runs), -min(r[1] for r in runs)


# ── Rolling per-EA stats ──────────────────────────────────────────────────────

def ea_stats(hist):
    """
    Per-EA stats over a trailing history window (DataFrame days x EAs).
    Returns DataFrame indexed by ea_id: total, mean, vol, sharpe, max_dd,
    loss_streak (consecutive losing non-zero days, counted from the end).
    """
    total = hist.sum()
    mean  = hist.mean()
    vol   = hist.std(ddof=0)
    sharpe = pd.Series(
        np.where(vol > 0, mean / vol * np.sqrt(TRADING_DAYS), 0.0),
        index=hist.columns)

    cum    = hist.cumsum()
    max_dd = cum.cummax().sub(cum).max()          # $ peak-to-trough per EA

    streaks = {}
    arr = hist.to_numpy()
    for j, ea in enumerate(hist.columns):
        col = arr[:, j]
        col = col[col != 0.0]                     # ignore no-trade days
        n = 0
        for v in col[::-1]:
            if v < 0:
                n += 1
            else:
                break
        streaks[ea] = n

    return pd.DataFrame({
        'total'      : total,
        'mean'       : mean,
        'vol'        : vol,
        'sharpe'     : sharpe,
        'max_dd'     : max_dd,
        'loss_streak': pd.Series(streaks),
    })


def rank_metric(stats, metric):
    """Series to rank EAs by (higher = better)."""
    if metric == 'return':
        return stats['total']
    if metric == 'calmar':
        return stats['total'] / stats['max_dd'].replace(0, np.nan)
    return stats['sharpe']                        # default


def correlations_vs(hist, candidate, members):
    """Max abs correlation of candidate vs current members over hist window."""
    if not members:
        return 0.0
    c = hist[candidate]
    if c.std(ddof=0) == 0:
        return 0.0
    best = 0.0
    for m in members:
        s = hist[m]
        if s.std(ddof=0) == 0:
            continue
        r = c.corr(s)
        if pd.notna(r):
            best = max(best, abs(r))
    return best


# ── Regimes ───────────────────────────────────────────────────────────────────
# A regime exposes review(date, hist, weights) -> (new_weights, [events]).
# hist contains only data up to the day BEFORE the review takes effect.

class EqualWeight:
    """Benchmark: hold every candidate at gross/N. Never changes."""
    name = 'equal_weight'

    def __init__(self, candidates, gross):
        self.candidates, self.gross = candidates, gross
        self._done = False

    def review(self, date, hist, weights):
        if self._done:
            return weights, []
        self._done = True
        w = pd.Series(0.0, index=weights.index)
        w[self.candidates] = self.gross / len(self.candidates)
        return w, [{'date': date, 'action': 'init',
                    'detail': f'equal weight {len(self.candidates)} EAs'}]


class RandomRotation:
    """Benchmark: same slots/cadence as rotation, dice-roll picks."""
    name = 'random'

    def __init__(self, candidates, gross, n_slots, seed=42):
        self.candidates, self.gross, self.n = candidates, gross, n_slots
        self.rng = np.random.default_rng(seed)

    def review(self, date, hist, weights):
        pick = list(self.rng.choice(self.candidates,
                                    size=min(self.n, len(self.candidates)),
                                    replace=False))
        w = pd.Series(0.0, index=weights.index)
        w[pick] = self.gross / len(pick)
        prev = set(weights[weights > 0].index)
        events = [{'date': date, 'action': 'swap', 'detail': f'random pick {len(pick)}'}] \
                 if prev != set(pick) else []
        return w, events


class StaticTopN:
    """Benchmark: rank once on the first review window, hold forever."""
    name = 'static_topn'

    def __init__(self, candidates, gross, n_slots, metric='sharpe'):
        self.candidates, self.gross, self.n, self.metric = candidates, gross, n_slots, metric
        self._done = False

    def review(self, date, hist, weights):
        if self._done:
            return weights, []
        self._done = True
        stats = ea_stats(hist[self.candidates])
        top = rank_metric(stats, self.metric).nlargest(self.n).index.tolist()
        w = pd.Series(0.0, index=weights.index)
        w[top] = self.gross / len(top)
        return w, [{'date': date, 'action': 'init',
                    'detail': f'static top {self.n} by {self.metric}: ' + ', '.join(top)}]


class Momentum:
    """Cross-sectional momentum: top-N by trailing metric, re-ranked each review."""
    name = 'momentum'

    def __init__(self, candidates, gross, n_slots, lookback=63, metric='sharpe',
                 max_per_symbol=None, streak_dollar_limit=None):
        self.candidates, self.gross, self.n = candidates, gross, n_slots
        self.lookback, self.metric = lookback, metric
        self.max_sym = max_per_symbol
        self.dollar_lim = streak_dollar_limit   # exclude robots mid-costly-streak

    def review(self, date, hist, weights):
        window = hist[self.candidates].tail(self.lookback)
        eligible = self.candidates
        if self.dollar_lim:
            eligible = [ea for ea in self.candidates
                        if -day_streak_stats(window[ea])[1] < self.dollar_lim]
        stats = ea_stats(window[eligible] if eligible else window)
        top = pick_top(stats, self.metric, self.n, self.max_sym)
        w = pd.Series(0.0, index=weights.index)
        w[top] = self.gross / len(top)
        prev = set(weights[weights > 0].index)
        events = []
        for ea in sorted(set(top) - prev):
            events.append({'date': date, 'action': 'add', 'ea_id': ea,
                           'detail': f'rank in top {self.n} by {self.metric}'})
        for ea in sorted(prev - set(top)):
            events.append({'date': date, 'action': 'drop', 'ea_id': ea,
                           'detail': f'fell out of top {self.n}'})
        return w, events


class InverseVol:
    """Risk parity (simple): weight every candidate proportional to 1/vol."""
    name = 'inverse_vol'

    def __init__(self, candidates, gross, lookback=63):
        self.candidates, self.gross, self.lookback = candidates, gross, lookback

    def review(self, date, hist, weights):
        vol = hist[self.candidates].tail(self.lookback).std(ddof=0)
        inv = 1.0 / vol.replace(0, np.nan)
        inv = inv.fillna(0.0)
        w = pd.Series(0.0, index=weights.index)
        if inv.sum() > 0:
            w[self.candidates] = inv / inv.sum() * self.gross
        return w, []


class Rules:
    """
    The configurable rotation regime: drop on loss streak / EA drawdown,
    refill from substitutes by trailing metric, optional correlation cap,
    cooldown after a drop.
    """
    name = 'rules'

    def __init__(self, portfolio, substitutes, gross, lookback=63,
                 metric='sharpe', loss_streak_limit=None, ea_dd_limit_pct=None,
                 corr_cap=None, cooldown_days=21, max_per_symbol=None,
                 streak_mode='days', streak_dollar_limit=None,
                 loss_count_limit=None, loss_count_window=21,
                 tradebook=None, relative_ratio=None, rel_baselines=None,
                 rel_expanding=False, basis=100_000,
                 capacity=None, fill_blanks_after=0,
                 pick_from_top=1, seed=42):
        self.active      = list(portfolio)
        self.subs        = list(substitutes)
        self.gross       = gross
        # capacity > len(portfolio) leaves BLANK slots: budgeted risk held in
        # reserve, which the refill may only use after fill_blanks_after
        # trading days (benched robots are still replaced immediately, up to
        # the starting team size).
        self.n_slots     = int(capacity) if capacity else len(portfolio)
        self.start_len   = len(portfolio)
        self.fill_after  = int(fill_blanks_after or 0)
        self._h0         = None
        # pick_from_top > 1: each refill chooses at random among the K best
        # eligible candidates instead of always the single best — spreads
        # promotions across near-equals. Seeded → reproducible.
        self.pick_top    = max(1, int(pick_from_top or 1))
        self._rng        = np.random.default_rng(seed)
        self.lookback    = lookback
        self.metric      = metric
        self.streak_lim  = loss_streak_limit
        self.dd_lim      = (ea_dd_limit_pct / 100.0 * basis) if ea_dd_limit_pct else None
        self.corr_cap    = corr_cap
        self.cooldown    = cooldown_days
        self.max_sym     = max_per_symbol
        self.streak_mode = streak_mode            # 'days' | 'trades'
        self.dollar_lim  = streak_dollar_limit    # $ lost over the current streak
        self.count_lim   = loss_count_limit       # N losses within loss_count_window
        self.count_win   = loss_count_window      # trading days
        self.tradebook   = tradebook              # required for 'trades' mode
        self.rel_ratio   = relative_ratio         # e.g. 1.0 = at historical worst
        self.rel_base    = rel_baselines or {}    # ea_id -> (streak_n, streak_$)
        self.rel_expand  = rel_expanding          # derive baselines in-sim
        self._benched    = {}                     # ea_id -> review date benched

    def streak_info(self, ea, window, asof):
        """(streak_len, streak_dollars, losses_in_count_window, unit_label)."""
        if self.streak_mode == 'trades' and self.tradebook is not None:
            n, dollars = self.tradebook.streak(ea, asof)
            cnt = 0
            if self.count_lim:
                cwin = window.tail(self.count_win)
                cnt = self.tradebook.losses_between(ea, cwin.index[0], asof)
            return n, dollars, cnt, 'trades'
        n, dollars = day_streak_stats(window[ea])
        cnt = 0
        if self.count_lim:
            cvals = window[ea].tail(self.count_win)
            cvals = cvals[cvals != 0]
            cnt = int((cvals < 0).sum())
        return n, dollars, cnt, 'days'

    def drop_reason(self, ea, window, stats, asof, hist=None):
        """First benching rule that fires for this EA, or None."""
        n, dollars, cnt, unit = self.streak_info(ea, window, asof)
        if self.streak_lim and n >= self.streak_lim:
            return f"loss streak {n} {unit} >= {self.streak_lim}"
        if self.dollar_lim and -dollars >= self.dollar_lim:
            return (f"current streak has lost ${-dollars:,.0f} "
                    f">= ${self.dollar_lim:,.0f}")
        if self.count_lim and cnt >= self.count_lim:
            return (f"{cnt} losing {unit} in last {self.count_win} "
                    f"trading days >= {self.count_lim}")
        if self.dd_lim and stats.at[ea, 'max_dd'] >= self.dd_lim:
            return f"window DD ${stats.at[ea, 'max_dd']:,.0f} >= ${self.dd_lim:,.0f}"

        # Relative rules: current streak vs this EA's own historical worst.
        # Fires only on a NEW record (strictly exceeding the baseline) that is
        # also >= ratio x baseline, so a baseline of 1 doesn't hair-trigger.
        if self.rel_ratio:
            if ea in self.rel_base:
                bn, bc = self.rel_base[ea]
            elif self.rel_expand and hist is not None:
                bn, bc = day_streak_baseline(hist[ea])
            else:
                bn = bc = 0
            if bn > 0 and n > bn and n >= self.rel_ratio * bn:
                return (f"streak {n} {unit} = {n / bn:.1f}x its historical "
                        f"worst ({bn})")
            if bc > 0 and -dollars > bc and -dollars >= self.rel_ratio * bc:
                return (f"streak cost ${-dollars:,.0f} = {-dollars / bc:.1f}x "
                        f"its historical worst (${bc:,.0f})")
        return None

    def review(self, date, hist, weights):
        window = hist.tail(self.lookback)
        asof   = hist.index[-1]
        pool   = sorted(set(self.active) | set(self.subs))
        stats  = ea_stats(window[pool])
        events = []

        # Blank-slot hold-back: refill target stays at the starting team size
        # until fill_blanks_after trading days have elapsed since first review.
        if self._h0 is None:
            self._h0 = len(hist)
        elapsed = len(hist) - self._h0
        cap = (self.n_slots if elapsed >= self.fill_after
               else min(self.start_len, self.n_slots))

        # 1. Drops
        for ea in list(self.active):
            reason = self.drop_reason(ea, window, stats, asof, hist=hist)
            if reason:
                self.active.remove(ea)
                self._benched[ea] = date
                events.append({'date': date, 'action': 'drop', 'ea_id': ea, 'detail': reason})

        # 2. Refills from substitutes (and recovered benched EAs)
        ranked = rank_metric(stats, self.metric).sort_values(ascending=False)
        logged = set()
        while len(self.active) < cap:
            # Collect the K best eligible candidates for this slot
            elig = []
            for ea in ranked.index:
                if ea in self.active:
                    continue
                benched_on = self._benched.get(ea)
                if benched_on is not None:
                    days_out = len(hist.loc[benched_on:])
                    if days_out < self.cooldown:
                        continue
                if self.max_sym:
                    sym    = symbol_of(ea)
                    n_same = sum(1 for a in self.active if symbol_of(a) == sym)
                    if n_same >= self.max_sym:
                        if ea not in logged:
                            logged.add(ea)
                            events.append({'date': date, 'action': 'reject', 'ea_id': ea,
                                           'detail': f'symbol cap: already {n_same} on {sym}'})
                        continue
                if self.corr_cap is not None:
                    c = correlations_vs(window, ea, self.active)
                    if c > self.corr_cap:
                        if ea not in logged:
                            logged.add(ea)
                            events.append({'date': date, 'action': 'reject', 'ea_id': ea,
                                           'detail': f'corr {c:.2f} > cap {self.corr_cap}'})
                        continue
                elig.append(ea)
                if len(elig) >= self.pick_top:
                    break
            if not elig:
                break
            if self.pick_top > 1 and len(elig) > 1:
                pick = elig[int(self._rng.integers(len(elig)))]
                why  = (f'random pick from top {len(elig)} by {self.metric} '
                        f'(ranked #{elig.index(pick) + 1})')
            else:
                pick = elig[0]
                why  = f'best available by {self.metric}'
            self.active.append(pick)
            self._benched.pop(pick, None)
            events.append({'date': date, 'action': 'add', 'ea_id': pick,
                           'detail': why})

        w = pd.Series(0.0, index=weights.index)
        if self.active:
            w[self.active] = self.gross / self.n_slots
        return w, events


REGIMES = {
    'equal_weight': EqualWeight,
    'random'      : RandomRotation,
    'static_topn' : StaticTopN,
    'momentum'    : Momentum,
    'inverse_vol' : InverseVol,
    'rules'       : Rules,
}


# ── Overlays (portfolio-level gross multiplier) ───────────────────────────────

class VolTarget:
    """Scale gross so trailing realized portfolio vol ~= an annualized $ target."""

    def __init__(self, target_ann_vol, lookback=21, min_mult=0.25, max_mult=2.0):
        self.target_daily = target_ann_vol / np.sqrt(TRADING_DAYS)
        self.lookback, self.min_mult, self.max_mult = lookback, min_mult, max_mult

    def multiplier(self, raw_pnl_hist, equity_hist, basis):
        if len(raw_pnl_hist) < self.lookback:
            return 1.0
        vol = np.std(raw_pnl_hist[-self.lookback:])
        if vol <= 0:
            return 1.0
        return float(np.clip(self.target_daily / vol, self.min_mult, self.max_mult))


class DDDerisk:
    """Cut gross as managed-equity drawdown deepens; restore on recovery."""

    def __init__(self, start_pct=3.0, floor_pct=6.0, min_mult=0.25):
        self.start, self.floor, self.min_mult = start_pct, floor_pct, min_mult

    def multiplier(self, raw_pnl_hist, equity_hist, basis):
        if not len(equity_hist):
            return 1.0
        peak = max(equity_hist)
        dd_pct = (peak - equity_hist[-1]) / basis * 100.0
        if dd_pct <= self.start:
            return 1.0
        if dd_pct >= self.floor:
            return self.min_mult
        frac = (dd_pct - self.start) / (self.floor - self.start)
        return float(1.0 - frac * (1.0 - self.min_mult))


OVERLAYS = {'vol_target': VolTarget, 'dd_derisk': DDDerisk}


# ── Simulation loop ───────────────────────────────────────────────────────────

def simulate(daily, regime, review_every=5, warmup=63, overlays=None, basis=100_000):
    """
    Walk the matrix. Reviews happen every `review_every` trading days starting
    after `warmup` days; each review sees history up to the prior day only.
    Returns dict: equity (DataFrame), weights (DataFrame), events, summary.
    """
    overlays  = overlays or []
    ea_index  = daily.columns
    weights   = pd.Series(0.0, index=ea_index)

    rows, weight_rows, events = [], [], []
    raw_hist, equity_hist = [], []
    equity = basis

    for t, date in enumerate(daily.index):
        if t >= warmup and (t - warmup) % review_every == 0:
            weights, evs = regime.review(date, daily.iloc[:t], weights)
            events.extend(evs)

        raw_pnl = float((daily.iloc[t] * weights).sum())

        mult = 1.0
        for ov in overlays:
            mult *= ov.multiplier(raw_hist, equity_hist, basis)

        pnl     = raw_pnl * mult
        equity += pnl
        raw_hist.append(raw_pnl)
        equity_hist.append(equity)

        rows.append({'date': date, 'raw_pnl': raw_pnl, 'gross_mult': mult,
                     'pnl': pnl, 'equity': equity})
        weight_rows.append(weights.rename(date))

    eq = pd.DataFrame(rows).set_index('date')
    wdf = pd.DataFrame(weight_rows)
    wdf.index.name = 'date'

    # ── Summary stats (post-warmup) ───────────────────────────────────────
    live   = eq.iloc[warmup:]
    pnl    = live['pnl']
    curve  = live['equity']
    peak   = curve.cummax()
    dd     = peak - curve
    years  = len(live) / TRADING_DAYS
    vol_d  = pnl.std(ddof=0)

    # Churn = weight changes AFTER the initial deployment — entering the
    # starting book on day one is not a management decision.
    dturn = wdf.diff().abs().sum(axis=1)
    nonzero = dturn[dturn > 0]
    entry = float(nonzero.iloc[0]) if len(nonzero) else 0.0
    turnover = (dturn.sum() - entry) / 2.0
    n_swaps  = len([e for e in events if e.get('action') in ('add', 'drop', 'swap')])

    summary = {
        'days'           : int(len(live)),
        'net_profit'     : round(float(pnl.sum()), 2),
        'ann_return_pct' : round(float(pnl.sum() / years / basis * 100), 2),
        'ann_vol_pct'    : round(float(vol_d * np.sqrt(TRADING_DAYS) / basis * 100), 2),
        'sharpe'         : round(float(pnl.mean() / vol_d * np.sqrt(TRADING_DAYS)), 2)
                           if vol_d > 0 else 0.0,
        'max_dd'         : round(float(dd.max()), 2),
        'max_dd_pct'     : round(float(dd.max() / basis * 100), 2),
        'calmar'         : round(float(pnl.sum() / years / dd.max()), 2) if dd.max() > 0 else 0.0,
        'turnover_units' : round(float(turnover), 1),
        'events'         : n_swaps,
    }
    return {'equity': eq, 'weights': wdf, 'events': events, 'summary': summary}
