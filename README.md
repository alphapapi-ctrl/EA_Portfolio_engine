# UBS Portfolio Manager

*(repo/codebase name: EA Portfolio Engine)*

A simulator and educational app for testing **portfolio management styles for
teams of trading robots (EAs)** against backtest history — before risking
anything live.

The question it answers: *does actively managing a team of EAs — benching
slumping robots, promoting in-form ones, capping concentration — actually beat
just letting everyone trade?* Humans are famously bad at this decision
(winning streaks breed overconfidence, losing streaks breed panic, systems get
shelved right before they recover). This engine replaces that emotional loop
with **written rules, tested against history** — including the honest control
tests that stop you fooling yourself.

## What's inside

**Streamlit app** (`app.py`, port 8504) — built as an educational tool for
people with no quant background; every control and metric is explained in
plain language:

- **Learn** — the whole concept, findings so far, and two evidence-backed
  starter configurations
- **Data** — compile timelines from MT5 Strategy Tester reports; refresh
  market reference data
- **EA Pool** — browse the robot pool, with a drawdown-calibration lie detector
- **Regimes** — where each robot's profit historically came from (dollar
  strong/weak, VIX calm/stressed, crypto bull/bear, …), descriptive not
  predictive
- **Build a Run** — configure a management style (rotation rules, streak
  rules in days/trades/dollars, diversification caps, risk scaling, safety
  overlays) and simulate it day by day with no look-ahead
- **Interactive Replay** — play through history; the sim pauses at every rule
  trigger and *you* make the call, with the evidence shown in words; rules can
  be changed mid-session and every decision is journaled
- **Results & Compare** — leaderboard with benchmarks, equity/drawdown
  charts, full decision journals

**Analysis scripts** (CLI): benchmark suite, parameter sweeps, seed-averaged
random controls, walk-forward validation, block-bootstrap Monte Carlo, review
cadence sweeps, and a market-regime conditioning matrix.

## The method

1. Every EA is backtested on a **fixed balance with fixed lots**, sized so its
   historical max drawdown ≈ 5% of the account. That makes P&L additive and
   linear in weight — any rotation/risk-scaling scheme can be simulated from
   the compiled daily P&L matrix without re-running backtests.
2. Regimes are asked for weights using **only history up to the previous
   day** — no look-ahead anywhere.
3. Every candidate style must beat three controls: **equal-weight
   hold-everything**, **random rotation** at the same cadence (seed-averaged),
   and a **static top-N**. It must also survive a parameter sweep (plateau,
   not spike), a walk-forward split (knobs frozen on old data), and Monte
   Carlo across block-reshuffled histories.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 1. drop MT5 Strategy Tester .htm reports into reports_in/  (subfolders = strategy families)
.venv\Scripts\python compile_timeline.py --name my_pool

# 2. (optional) market data for the Regimes page
.venv\Scripts\python fetch_reference_data.py
.venv\Scripts\python build_regime_matrix.py --timeline my_pool

# 3. the app
.venv\Scripts\streamlit run app.py          # http://localhost:8504

# or the CLI suite
.venv\Scripts\python run_sim.py --preset all --timeline my_pool
```

On Windows, `launch.bat` starts (or focuses) the app with a double-click.

## What's committed vs local

The **compiled timeline database** (`timeline/` — aggregated trade lists,
daily P&L matrices, and metadata) is included so the app and analysis scripts
work out of the box. The **raw inputs stay local and git-ignored**:
`reports_in/` (Strategy Tester reports / set files), `runs/` (your results),
and `reference/` (market data, refreshable from Yahoo Finance in one click
on the Data page).

## Working with an AI assistant

If you're using ChatGPT, Claude, or similar to analyse this data or extend
the engine, point it at **[AI_GUIDE.md](AI_GUIDE.md)** first — it documents
the repo structure, data schemas, the risk-normalisation semantics, the
invariants (no look-ahead, survivorship caveats), and the findings already
established, so the assistant starts productive instead of re-deriving basics.

## Honest caveats

- Backtest pools are **survivors** — every robot is there because its history
  looked good, which inflates every absolute number. Only comparisons
  *between* styles are meaningful.
- Historical max drawdown is **not a limit**; the Monte Carlo tooling exists
  precisely to quantify what "normal bad" and "unlucky bad" look like before
  they happen.
- Nothing here is financial advice. It's a research and education tool.
