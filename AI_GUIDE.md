# AI Guide — how to work with this repo

This file is written for an AI assistant (ChatGPT, Claude, etc.) helping a
human use or extend the EA Portfolio Engine. Read this first; it defines the
data semantics, the invariants you must not break, and what has already been
established so you don't re-derive or contradict it.

## What this is

A simulator for **portfolio management styles over a pool of trading robots
(EAs)**, driven by compiled MT5 backtest data. The human picks a management
style (rotation rules, momentum ranking, risk scaling…) and the engine
replays 2020–2026 day by day to show what that style would have done.
There is also a Streamlit app (`app.py`, port 8504) built for novices —
if you add features to it, every control and metric must carry a
plain-language explanation; that is a hard product requirement.

## Repo map

| File | Role |
|---|---|
| `compile_timeline.py` | Parses MT5 Strategy Tester `.htm` reports → a timeline dataset under `timeline/<name>/` |
| `parsers.py` | Report parsing (trades via FIFO deal-pairing, summary, input parameters) |
| `portfolio_sim.py` | **Core.** Timeline loading, rolling stats, regimes, overlays, `simulate()` loop |
| `interactive_sim.py` | Pausable session for the app's Interactive Replay (proposals → human decisions) |
| `run_sim.py` | Run configs (JSON) → results under `runs/<name>/`; preset benchmark suite |
| `sweep_*.py`, `walk_forward*.py`, `monte_carlo*.py` | Robustness batteries (see "Established findings") |
| `fetch_reference_data.py` | Yahoo Finance daily series (regime indicators + EA markets) |
| `build_regime_matrix.py` | Conditions each EA's daily P&L on market regime states |
| `app.py` | The Streamlit app (Learn / Data / EA Pool / Regimes / Build a Run / Interactive Replay / Results) |

## The data model (read in this order)

1. **`timeline/main_pool/manifest.json`** — self-describing summary: dataset
   stats, risk normalisation, file schemas, and per-EA metadata as JSON.
   Start here; it is small and answers most questions.
2. **`timeline/main_pool/ea_meta.csv`** — one row per EA: `ea_id` (unique key),
   `strategy` (set-file stem), `family` (source folder ≈ strategy suite),
   `symbol`, `timeframe`, `hist_max_dd`, `lot_step`, realized backtest DD,
   `dd_vs_target`, trade counts and date range.
3. **`timeline/main_pool/daily_pnl.csv`** — the core matrix: one row per date,
   one column per `ea_id`, value = net P&L ($) of trades closed that day.
   0 = no trades closed. ~1,820 rows × 130 columns.
4. **`timeline/main_pool/trades.csv`** — every trade (24MB). Only needed for
   trade-level analysis (e.g. streaks counted in trades).
5. **`timeline/main_pool/regime_states.csv` / `regime_matrix.csv`** — daily
   market regime states and each EA's performance conditioned on them.

## Semantics you must preserve

- **Weights are risk units.** Every backtest ran on a fixed $100k balance with
  fixed lots calibrated so the EA's *historical* max drawdown ≈ 5% of balance
  ($5,000). Therefore: weight 1.0 = the EA as backtested; P&L is **linear**
  (weight w ⇒ w × daily P&L) and **additive** across EAs; there is **no
  compounding** anywhere. Any resizing scheme is just arithmetic on the
  matrix — never re-run backtests to rescale.
- **`hist_max_dd` is a calibration input, not a limit.** Future drawdowns can
  and do exceed it. `dd_vs_target` far from 1.0 marks EAs whose set-file
  calibration was stale or a default (common in third-party sets). Warn, don't
  silently "fix".
- **No look-ahead.** A regime's `review(date, hist, weights)` receives history
  strictly up to the *previous* day. Any new rule, metric, or selection step
  must respect this. This is the most important invariant in the codebase.
- **The pool is survivors.** Every EA is here because its backtest looked
  good. Absolute performance numbers (Sharpe 8–9!) are inflated artifacts.
  **Only comparisons between styles on the same pool are meaningful.** Never
  present absolute figures as expectations.

## How to run things

```bash
# compile a dataset from MT5 tester reports (UTF-16 .htm files)
python compile_timeline.py --reports <folder> --name my_pool

# benchmark + regime suite, comparison table
python run_sim.py --preset all --timeline my_pool
python run_sim.py --compare

# custom run: JSON config (full schema documented at top of run_sim.py)
python run_sim.py --config runs/my_run.json

# robustness batteries
python sweep_analysis.py            # parameter grids + random-seed controls
python walk_forward.py [--review-every N]   # calibrate pre-2023, frozen test after
python monte_carlo.py               # block-bootstrap across reshuffled histories
```

Run config essentials: `regime` ∈ equal_weight | random | static_topn |
momentum | inverse_vol | rules; `portfolio` = list of ea_ids, `"all"`, or
`"top10_sharpe"`; rules params include `loss_streak_limit`, `streak_mode`
('days'|'trades'), `streak_dollar_limit`, `loss_count_limit`+window,
`ea_dd_limit_pct`, `corr_cap`, `cooldown_days`, `max_per_symbol`;
`start_date`/`end_date` slice the test period; `gross_budget` sets total risk
units deployed.

## Extending

- **New regime**: a class with `review(date, hist, weights) -> (weights, events)`
  where `weights` is a pd.Series over all EA columns (risk units). Register in
  `portfolio_sim.REGIMES` and wire params in `run_sim.build_regime`. Events are
  dicts (`date`, `action`, `ea_id`, `detail`) — keep `detail` human-readable;
  it is the audit journal shown to users.
- **New overlay**: class with `multiplier(raw_pnl_hist, equity_hist, basis) -> float`,
  register in `OVERLAYS`.
- **Any new rule must be swept** (is the parameter surface a plateau or a
  spike?) and ideally walk-forwarded before being presented as a finding.

## Established findings (don't re-litigate without new evidence)

Tested via benchmark controls, seed-averaged random, parameter sweeps,
walk-forward (calibrate 2020–22 / frozen test 2023–26), and block-bootstrap
Monte Carlo (300 reshuffled histories):

1. Drop-on-damage **rules regimes beat momentum ranking** out-of-sample, and
   both crush random rotation (which proves the rankings carry signal).
2. **Equal weight is the honest benchmark** — lowest DD everywhere; beat it
   per unit of risk or the style adds nothing.
3. **Concentration is the main trap**: top-N selection by Sharpe on 2020–22
   yields 9/10 Bitcoin EAs. `max_per_symbol` (diversification cap) fixes
   selection; the rules regime also self-repairs over time.
4. A single **streak-cost rule** ("bench when the current losing streak has
   cost ≥ $1,000") with ~3-day reviews matched or beat multi-rule setups
   out-of-sample with ~25 decisions in 3.5 years. The multi-rule setup (add
   `ea_dd_limit_pct` ≈ 2–2.5%) gives the smallest drawdowns.
5. **Review cadence**: with rules, frequent review (1–5 days) adds safety
   without churn; slow review (10–21 days) is the only harmful zone. With
   momentum, cadence changes workload, not results.
6. Monte Carlo DD expectations for a 10-slot rules book: median worst drop
   ≈ 5%, 95th percentile ≈ 8%. These are "normal bad" and "unlucky bad" —
   quantified so a drawdown doesn't get misread as "the system broke".
7. The regime matrix (Regimes page) is **descriptive, not predictive** —
   states are coincident. Present it as context, never as a signal, unless a
   proper simulated test says otherwise.

## Gotchas

- MT5 set files and tester reports are **UTF-16 LE with BOM**; `parsers._decode`
  handles it — don't read them as UTF-8.
- Streamlit markdown treats paired `$` as LaTeX — escape as `\$` or write
  "dollars" (bugs from this have already been fixed twice).
- Yahoo Finance: use `yf.download(period='max')` then slice; `start=`/`end=`
  intermittently returns near-empty data for index tickers (^TNX). Fallbacks:
  Dukascopy for instruments, FRED DGS10 for yields.
- `reports_in/`, `runs/`, `reference/` are git-ignored on purpose (private
  raw data / regenerable). The committed `timeline/main_pool/` is the shared
  working dataset.
