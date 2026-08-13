"""
monte_carlo_streak_cost.py
==========================
Block-bootstrap Monte Carlo for the streak-cost-only rules regime
(bench when the current losing streak has cost >= $1,000; no other benching
rules; corr cap 0.7). Reuses the monte_carlo harness with the same seeds and
block sizes so results are directly comparable to the original run.

Outputs to runs/_monte_carlo_streak_cost/.
"""

import os

import monte_carlo as mc

mc.RULES_PARAMS = dict(lookback=63, metric='sharpe',
                       loss_streak_limit=None, ea_dd_limit_pct=None,
                       streak_dollar_limit=1000,
                       corr_cap=0.7, cooldown_days=21)
mc.OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'runs', '_monte_carlo_streak_cost')

if __name__ == '__main__':
    mc.main()
