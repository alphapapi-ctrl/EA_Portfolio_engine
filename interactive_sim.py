"""
interactive_sim.py
==================
EA Portfolio Engine — interactive (play/pause) simulation session.

Runs the rules regime day by day like portfolio_sim.simulate, but instead of
applying rotation decisions automatically it PAUSES at each review that
produces proposals and waits for the user to approve, modify, or reject each
one. Built for the Streamlit replay page; no Streamlit imports here.

Each proposal is a swap suggestion:
    {'drop': ea_id or None, 'drop_reason': str,
     'add': ea_id or None,  'add_reason': str, 'evidence': {...}}
"""

import numpy as np
import pandas as pd

from portfolio_sim import (ea_stats, rank_metric, correlations_vs, symbol_of,
                           day_streak_stats, TRADING_DAYS)


class InteractiveSession:

    def __init__(self, daily, portfolio, substitutes, gross=10.0,
                 review_every=5, warmup=63, lookback=63, metric='sharpe',
                 loss_streak_limit=5, ea_dd_limit_pct=2.5,
                 corr_cap=None, cooldown_days=21, max_per_symbol=None,
                 streak_mode='days', streak_dollar_limit=None,
                 loss_count_limit=None, loss_count_window=21, tradebook=None,
                 basis=100_000):
        self.daily        = daily
        self.gross        = gross
        self.review_every = review_every
        self.warmup       = warmup
        self.lookback     = lookback
        self.metric       = metric
        self.streak_lim   = loss_streak_limit
        self.dd_lim       = (ea_dd_limit_pct / 100.0 * basis) if ea_dd_limit_pct else None
        self.corr_cap     = corr_cap
        self.cooldown     = cooldown_days
        self.max_sym      = max_per_symbol
        self.streak_mode  = streak_mode
        self.dollar_lim   = streak_dollar_limit
        self.count_lim    = loss_count_limit
        self.count_win    = loss_count_window
        self.tradebook    = tradebook
        self.basis        = basis

        self.active    = list(portfolio)
        self.n_slots   = len(portfolio)
        self.subs      = [s for s in substitutes if s not in self.active]
        self._benched  = {}                      # ea_id -> day index benched

        self.t         = 0                       # next day to process
        self.equity    = float(basis)
        self.weights   = pd.Series(0.0, index=daily.columns)
        self._set_weights()
        self.equity_rows = []                    # per-day dicts
        self.journal     = []                    # decision journal
        self.pending     = None                  # proposals awaiting the user
        self.pending_date = None

    # ── internals ─────────────────────────────────────────────────────────

    def _set_weights(self):
        w = pd.Series(0.0, index=self.daily.columns)
        if self.active:
            w[self.active] = self.gross / self.n_slots
        self.weights = w

    def _step_day(self):
        date = self.daily.index[self.t]
        pnl  = float((self.daily.iloc[self.t] * self.weights).sum())
        self.equity += pnl
        self.equity_rows.append({'date': date, 'pnl': pnl, 'equity': self.equity,
                                 'n_active': len(self.active)})
        self.t += 1

    def _is_review_day(self):
        return self.t >= self.warmup and (self.t - self.warmup) % self.review_every == 0

    def _streak_info(self, ea, window, asof):
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
        return n, dollars, cnt, 'trading days'

    def _drop_reason(self, ea, window, stats, asof):
        n, dollars, cnt, unit = self._streak_info(ea, window, asof)
        if self.streak_lim and n >= self.streak_lim:
            return (f"has {n} losing {unit} in a row "
                    f"(your limit is {self.streak_lim})")
        if self.dollar_lim and -dollars >= self.dollar_lim:
            return (f"its current losing streak has cost ${-dollars:,.0f} "
                    f"(your limit is ${self.dollar_lim:,.0f})")
        if self.count_lim and cnt >= self.count_lim:
            return (f"has {cnt} losing {unit} within the last {self.count_win} "
                    f"trading days (your limit is {self.count_lim})")
        if self.dd_lim and stats.at[ea, 'max_dd'] >= self.dd_lim:
            return (f"has dropped ${stats.at[ea, 'max_dd']:,.0f} from its recent "
                    f"peak (your limit is ${self.dd_lim:,.0f})")
        return None

    def update_rules(self, **kw):
        """
        Mid-session rule changes — take effect from the next review.
        Accepts: loss_streak_limit, streak_dollar_limit, loss_count_limit,
        loss_count_window, streak_mode, ea_dd_limit_pct, corr_cap,
        cooldown_days, max_per_symbol. Logs the change to the journal.
        """
        mapping = {
            'loss_streak_limit'  : 'streak_lim',
            'streak_dollar_limit': 'dollar_lim',
            'loss_count_limit'   : 'count_lim',
            'loss_count_window'  : 'count_win',
            'streak_mode'        : 'streak_mode',
            'corr_cap'           : 'corr_cap',
            'cooldown_days'      : 'cooldown',
            'max_per_symbol'     : 'max_sym',
        }
        changed = []
        for key, val in kw.items():
            if key == 'ea_dd_limit_pct':
                new = (val / 100.0 * self.basis) if val else None
                if new != self.dd_lim:
                    self.dd_lim = new
                    changed.append(f"ea_dd_limit_pct={val}")
            elif key in mapping:
                if getattr(self, mapping[key]) != val:
                    setattr(self, mapping[key], val)
                    changed.append(f"{key}={val}")
        if changed:
            date = self.daily.index[min(self.t, len(self.daily) - 1)]
            self.journal.append({'date': date, 'choice': 'rule_change',
                                 'drop': None,
                                 'drop_reason': 'rules changed: ' + ', '.join(changed),
                                 'add': None, 'add_reason': ''})
        return changed

    def _build_proposals(self):
        hist   = self.daily.iloc[:self.t]
        window = hist.tail(self.lookback)
        asof   = hist.index[-1]
        pool   = sorted(set(self.active) | set(self.subs))
        stats  = ea_stats(window[pool])

        drops = []
        for ea in self.active:
            reason = self._drop_reason(ea, window, stats, asof)
            if reason:
                drops.append((ea, reason))

        vacancies = len(drops) + (self.n_slots - len(self.active))
        if vacancies == 0:
            return []

        # Candidate replacements, best first, respecting cooldown + corr cap
        remaining = [ea for ea in self.active if ea not in [d[0] for d in drops]]
        ranked    = rank_metric(stats, self.metric).sort_values(ascending=False)
        adds = []
        for ea in ranked.index:
            if len(adds) >= vacancies:
                break
            if ea in self.active or ea in [a[0] for a in adds]:
                continue
            benched_at = self._benched.get(ea)
            if benched_at is not None and (self.t - benched_at) < self.cooldown:
                continue
            add_reason = (f"best available candidate by {self.metric} "
                          f"over the last {self.lookback} trading days")
            if self.max_sym:
                sym    = symbol_of(ea)
                n_same = sum(1 for x in remaining + [a[0] for a in adds]
                             if symbol_of(x) == sym)
                if n_same >= self.max_sym:
                    continue
                add_reason += (f" (market check: {n_same} other on {sym}, "
                               f"cap is {self.max_sym})")
            if self.corr_cap is not None:
                c = correlations_vs(window, ea, remaining)
                if c > self.corr_cap:
                    continue
                add_reason += f" (correlation vs current team {c:.2f}, under your {self.corr_cap} cap)"
            adds.append((ea, add_reason))

        proposals = []
        for i in range(max(len(drops), len(adds))):
            drop = drops[i] if i < len(drops) else (None, '')
            add  = adds[i]  if i < len(adds)  else (None, '')
            ev   = {}
            for label, ea in (('drop', drop[0]), ('add', add[0])):
                if ea is None:
                    continue
                ev[label] = {
                    'window_pnl'  : round(float(stats.at[ea, 'total']), 2),
                    'window_dd'   : round(float(stats.at[ea, 'max_dd']), 2),
                    'loss_streak' : int(stats.at[ea, 'loss_streak']),
                    'sharpe'      : round(float(stats.at[ea, 'sharpe']), 2),
                }
            proposals.append({'drop': drop[0], 'drop_reason': drop[1],
                              'add': add[0], 'add_reason': add[1],
                              'evidence': ev})
        return proposals

    # ── public API ────────────────────────────────────────────────────────

    def advance(self):
        """
        Process days until the next review that produces proposals (pause),
        or the end of history. Returns 'paused' or 'done'.
        """
        if self.pending:
            return 'paused'
        while self.t < len(self.daily):
            if self._is_review_day():
                proposals = self._build_proposals()
                if proposals:
                    self.pending      = proposals
                    self.pending_date = self.daily.index[self.t]
                    return 'paused'
            self._step_day()
        return 'done'

    def apply_decisions(self, decisions):
        """
        decisions: list aligned with self.pending, each 'swap' | 'drop_only'
        | 'keep'. Applies them, logs to the journal, clears the pause.
        """
        date = self.pending_date
        for prop, choice in zip(self.pending, decisions):
            entry = {'date': date, 'choice': choice, **{k: prop[k] for k in
                     ('drop', 'drop_reason', 'add', 'add_reason')}}
            if choice in ('swap', 'drop_only') and prop['drop']:
                if prop['drop'] in self.active:
                    self.active.remove(prop['drop'])
                    self._benched[prop['drop']] = self.t
            if choice == 'swap' and prop['add']:
                if len(self.active) < self.n_slots:
                    self.active.append(prop['add'])
                    self._benched.pop(prop['add'], None)
            self.journal.append(entry)
        self._set_weights()
        self.pending      = None
        self.pending_date = None

    def equity_frame(self):
        if not self.equity_rows:
            return pd.DataFrame(columns=['date', 'pnl', 'equity', 'n_active'])
        return pd.DataFrame(self.equity_rows).set_index('date')

    def summary(self):
        eq = self.equity_frame()
        live = eq.iloc[self.warmup:] if len(eq) > self.warmup else eq
        if live.empty or live['pnl'].std(ddof=0) == 0:
            return {}
        pnl   = live['pnl']
        curve = live['equity']
        dd    = (curve.cummax() - curve)
        years = max(len(live) / TRADING_DAYS, 1e-9)
        return {
            'net_profit'     : round(float(pnl.sum()), 2),
            'ann_return_pct' : round(float(pnl.sum() / years / self.basis * 100), 2),
            'sharpe'         : round(float(pnl.mean() / pnl.std(ddof=0) * np.sqrt(TRADING_DAYS)), 2),
            'max_dd'         : round(float(dd.max()), 2),
            'max_dd_pct'     : round(float(dd.max() / self.basis * 100), 2),
            'decisions'      : len(self.journal),
        }
