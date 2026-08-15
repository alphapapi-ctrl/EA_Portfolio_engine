"""
EA Portfolio Engine — Streamlit app.
====================================
An educational tool for exploring EA portfolio management regimes on
backtested data. Built for people with no quant background: every control
and metric is explained in plain language.

Run:  streamlit run app.py   (port 8504 via .streamlit/config.toml)
"""

import os
import sys
import json
import glob
import shutil

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from portfolio_sim import load_timeline, load_trades, ea_stats, rank_metric, pick_top
from run_sim import run_one
from interactive_sim import InteractiveSession

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR   = os.path.join(ENGINE_DIR, 'runs')

st.set_page_config(page_title='EA Portfolio Engine', page_icon='⚙️',
                   layout='wide', initial_sidebar_state='expanded')


# ── Shared helpers ────────────────────────────────────────────────────────────

def list_timelines():
    base = os.path.join(ENGINE_DIR, 'timeline')
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base)
                  if os.path.isfile(os.path.join(base, d, 'daily_pnl.csv')))


def _desc_path(name):
    return os.path.join(ENGINE_DIR, 'timeline', name, 'description.txt')


def read_desc(name):
    p = _desc_path(name)
    if os.path.isfile(p):
        with open(p, encoding='utf-8') as f:
            return f.read().strip()
    return ''


def write_desc(name, text):
    with open(_desc_path(name), 'w', encoding='utf-8') as f:
        f.write(text.strip() + '\n')


SUITES_PATH = os.path.join(ENGINE_DIR, 'packaged_suites.json')


def load_suites():
    """Packaged-EA suite definitions (ships with the repo; see file _readme)."""
    if not os.path.isfile(SUITES_PATH):
        return []
    try:
        with open(SUITES_PATH, encoding='utf-8') as f:
            return [s for s in json.load(f).get('suites', [])
                    if isinstance(s, dict) and s.get('name')]
    except (json.JSONDecodeError, OSError):
        return []


def _sync_dataset(widget_key):
    st.session_state['dataset'] = st.session_state[widget_key]


def dataset_selector(widget_key, show_desc=False):
    """Render a dataset selectbox synced across the sidebar and pages.

    The chosen name lives in st.session_state['dataset']; each rendered
    widget gets its own key and writes back through _sync_dataset, so the
    sidebar copy and an on-page copy always agree.
    """
    timelines = list_timelines()
    if not timelines:
        st.warning('No timelines found. Compile one on the 🗂 Data page.')
        return None
    cur = st.session_state.get('dataset')
    if cur not in timelines:
        cur = next((p for p in ('main_pool_2018', 'main_pool')
                    if p in timelines), timelines[0])
    st.session_state['dataset'] = cur
    # Keyed widgets ignore index= once they hold state, so push the canonical
    # choice into the widget's own state before rendering.
    st.session_state[widget_key] = cur
    st.selectbox(
        'Dataset (timeline)', timelines,
        key=widget_key, on_change=_sync_dataset, args=(widget_key,),
        help='A timeline is a compiled bundle of backtests. Build, describe '
             'or delete them on the 🗂 Data page.')
    name = st.session_state['dataset']
    if show_desc:
        d = read_desc(name)
        if d:
            st.caption(f'📂 {d}')
        else:
            st.caption('📂 No description yet — add one on the 🗂 Data page.')
    return name


@st.cache_data(show_spinner=False)
def cached_timeline(name):
    return load_timeline(name)


@st.cache_resource(show_spinner=False)
def cached_tradebook(name):
    return load_trades(name)


def friendly_name(ea_id, meta):
    row = meta[meta.ea_id == ea_id]
    if row.empty:
        return ea_id
    r = row.iloc[0]
    return f"{r['strategy']}  ({r['symbol']} {r['timeframe']}, {r['family']})"


def equity_chart(frames, basis=100_000):
    """frames: {label: DataFrame with equity column indexed by date}"""
    fig = go.Figure()
    for label, df in frames.items():
        fig.add_trace(go.Scatter(x=df.index, y=df['equity'],
                                 mode='lines', name=label))
    fig.add_hline(y=basis, line_dash='dot', line_color='gray',
                  annotation_text='starting balance')
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation='h', y=-0.15),
                      yaxis_title='Account value ($)')
    return fig


def friendly_wf_table(df):
    """Walk-forward comparison with plain-language headers."""
    out = df.rename(columns={
        'run': 'Management style', 'net_profit': 'Profit ($)',
        'ann_return_pct': 'Per year (%)', 'sharpe': 'Sharpe',
        'max_dd_pct': 'Worst drop (%)', 'turnover_units': 'Churn',
        'events': 'Decisions'})
    keep = ['Management style', 'Profit ($)', 'Per year (%)', 'Sharpe',
            'Worst drop (%)', 'Churn', 'Decisions']
    return out[[c for c in keep if c in out.columns]]


def friendly_mc_table(df):
    """Monte Carlo summary with plain-language headers and values."""
    out = df.copy()
    out['block']  = out['block'].map({
        'weekly_5d': 'Weekly (5-day) blocks',
        'monthly_21d': 'Monthly (21-day) blocks',
        'quarterly_63d': 'Quarterly (63-day) blocks'}).fillna(out['block'])
    out['regime'] = out['regime'].map({
        'rules': 'Rules style', 'equal_weight': 'Equal weight'}).fillna(out['regime'])
    for col in ['P(dd>5%)', 'P(dd>10%)', 'P(loss)']:
        if col in out.columns:
            out[col] = (out[col] * 100).round(0).astype(int).astype(str) + '%'
    out = out.rename(columns={
        'block'     : 'History reshuffled in',
        'regime'    : 'Style',
        'sharpe_p05': 'Sharpe — unlucky end (5th percentile)',
        'sharpe_med': 'Sharpe — typical (median)',
        'sharpe_p95': 'Sharpe — lucky end (95th percentile)',
        'dd_pct_p05': 'Worst drop % — lucky end (5th percentile)',
        'dd_pct_med': 'Worst drop % — typical (median)',
        'dd_pct_p95': 'Worst drop % — unlucky end (95th percentile)',
        'net_med'   : 'Profit ($) — typical (median)',
        'P(dd>5%)'  : 'Chance of a drop over 5%',
        'P(dd>10%)' : 'Chance of a drop over 10%',
        'P(loss)'   : 'Chance of finishing at a loss'})
    return out


METRIC_HELP = {
    'net_profit'    : 'Total dollars made over the test (on the fixed $100k base).',
    'ann_return_pct': 'Average profit per year, as % of the $100k base.',
    'sharpe'        : ('Return per unit of "wobble". Higher = smoother ride for the '
                      'same profit. Above ~1 is decent in live trading; the huge '
                      'values here are inflated because this pool only contains '
                      'strategies that already looked good historically.'),
    'max_dd_pct'    : ('Biggest peak-to-valley drop of the account, as % of $100k. '
                      'The "how bad did it get" number.'),
    'turnover_units': 'How much swapping the regime did. Higher = more churn.',
}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('## ⚙️ EA Portfolio Engine')
    st.caption('Test *management styles* for a team of trading robots — '
               'on history, before risking anything.')
    page = st.radio('Page', ['📚 Learn', '🗂 Data', '📊 EA Pool', '🌐 Regimes',
                             '🛠 Build a Run', '▶ Interactive Replay',
                             '🏁 Results & Compare'],
                    label_visibility='collapsed')
    st.divider()
    timeline_name = dataset_selector('dataset_sidebar')


# ══════════════════════════════════════════════════════════════════════════════
# 📚 LEARN
# ══════════════════════════════════════════════════════════════════════════════

if page == '📚 Learn':
    st.title('📚 What is this tool?')

    st.markdown("""
Imagine you manage a **team of trading robots** (called *EAs* — Expert Advisors).
Each one trades its own market its own way. Some are on form, some are in a slump —
and form changes. The question this tool answers:

> **Does actively managing the team — benching slumping robots, promoting in-form
> ones — actually beat just letting everyone play?**

Humans are famously bad at this decision. A winning streak breeds overconfidence
(risk up, trade more), then a losing streak hits harder than it should, panic sets
in, the system gets shelved… usually right before it recovers. This tool replaces
that emotional loop with **written rules**, then *tests those rules against history*
so you can see what they would have done — before any real money is involved.
""")

    st.subheader('The dataset, in plain words')
    st.markdown("""
Every robot in the pool was backtested on the **same account size (100,000 dollars)**
with its trade size fixed so that its *worst historical losing stretch* equals
roughly **5,000 dollars (5%)**. That means every robot carries the **same risk weight** — one unit
of any robot risks about as much as one unit of any other. This makes team
comparisons fair, and it means "run robot X at half size" is exactly half the P&L.
""")
    with st.expander('⚠️ Two honest warnings about this data'):
        st.markdown("""
1. **Historical drawdown is not a limit.** The 5% calibration comes from each
   robot's *past* worst stretch. The future can (and sometimes will) be worse.
   Our simulations across shuffled histories say a ~5% team drawdown is *normal*
   and ~9% is unlucky-but-ordinary — neither means the system is broken.
2. **This pool is made of survivors.** Every robot here exists because its
   backtest looked good. That inflates every absolute number you'll see.
   **Compare regimes against each other — don't trust the raw profit figures.**
""")

    st.subheader('The management styles you can test')
    st.markdown("""
| Style | In plain words |
|---|---|
| **Equal weight** | Everyone plays, same size, never change anything. The humble benchmark — famously hard to beat. |
| **Static top-N** | Pick the N best-looking robots once, never touch them again. Set-and-forget. |
| **Momentum** | Every review, re-rank everyone by recent form and field the current top N. The "performance chaser". |
| **Rules (rotation)** | Only bench a robot when it breaks a written rule (losing streak too long, drawdown too deep), then promote the best available substitute. The "disciplined manager". |
| **Inverse volatility** | Everyone plays, but calmer robots get more size and wilder robots get less. No benching — just automatic size trimming. |
| **Random** | Swap robots by dice roll. Sounds silly — it's the control test. Any style worth using must beat random *by a lot*. |
""")

    st.subheader('What we found on this dataset')
    st.caption('All numbers below come from the full test battery re-run on the '
               'extended 2018–2026 window (140 robots, tick-verified recent '
               'data). Where the longer, harder window *changed* a conclusion, '
               'we say so — that is the test suite doing its job.')
    st.markdown("""
- **The headline: how widely you diversify matters more than how cleverly you
  manage — but at realistic size, management wins big.** Out-of-sample
  (knobs frozen on 2018–22, tested 2023–26), holding *all 140 robots* equally
  beat every management style per unit of risk (Sharpe 9.5, worst drop 1.8%) —
  but nobody actually runs 140 robots. At sizes people really trade, the
  picture flips hard: a passively-held random bucket of **10** robots averaged
  Sharpe **4.0** with an 8% worst drop — barely better than swapping robots at
  random — while the rules style managing 10 slots scored **8.0 with a 4.5%
  drop**: roughly double the smoothness at half the pain. Even the best
  passive bucket size tested (20 robots, Sharpe ~7.6) trailed every managed
  style. The honest hierarchy: **diversify as widely as you can genuinely
  operate; manage whatever concentration remains with written rules.** Breadth
  beats management, but management beats same-size do-nothing by a wide margin.
- **The synthesis, proven on a real portfolio.** We took an actual
  human-curated selection (37 robots, deliberately gold-heavy) and ran it both
  ways at the same risk budget. Held as-is, out-of-sample: Sharpe 6.3, worst
  drop 4.2% — respectable, beats small random buckets. The *same portfolio*
  under the written rules (with the full pool as its substitute bench):
  **Sharpe 10.2, worst drop 3.1%, nearly double the return** — the best
  out-of-sample result of anything tested, including the 140-robot spread.
  Breadth times rules beats either alone. Curation chooses the squad;
  rules coach it.
- **What management still wins: drawdown control and decision economy.** Among
  managed styles, the multi-rule setup had by far the smallest worst drop
  (3.0–4.5% vs 7.5–9.6% for momentum and set-and-forget) and the best
  return-per-worst-drop (Calmar ~60–73, the highest of anything tested) — with
  ~30–40 team changes in 3.5 years versus momentum's ~900. If you must
  concentrate, written rules remain the safest way to do it.
- **Random is still terrible** (Sharpe ~5 vs 8–9.5 for everything sensible,
  across 10 seeds) — the rankings and rules carry real signal; churn alone
  does not.
- **A concentrated team is still the big trap.** Ranking by past performance
  again picked 9 Bitcoin robots out of 10 — and holding that team untouched
  drew down 9.6% out-of-sample (11.9% in the full-window test). On this longer
  window the correlation cap went from "barely matters" to clearly valuable:
  in-sample it added ~0.7 Sharpe to the rules style. Diversify by rule, not
  by hope.
""")

    st.subheader('First steps to building a management system')
    st.caption('The findings above are what history says. These four steps turn '
               'them into a working system — in this order, each decided '
               '*before* any money moves.')
    st.markdown("""
**Step 1 — Set your expectations in writing, before you start.** Monte Carlo
across 300 reshuffled histories says a 10-slot rules book runs a median worst
drop of ~5.3%, ~9% at the unlucky 95th percentile, and exceeded 10% in at most
3% of histories (a wide equal-weight book: 1.8–3.1% in every single history).
Write down, today, which number means "normal", which means "unlucky", and
which means "broken — stop". That pre-made decision is what breaks the
panic-and-shelve cycle, because in the middle of a drawdown you will not be
able to make it calmly.

**Step 2 — Pick your rules (fewer, firmer).** One firm rule can carry the job:
"bench any robot whose current losing streak has cost 1,000 dollars" was
sweep-stable as the *only* rule, matched the multi-rule Monte Carlo range with
a tighter bad tail, and out-of-sample earned a slightly higher Sharpe than the
full setup with only ~20 decisions — at the price of a deeper worst drop
(7.6% vs 4.5%). Add the per-robot drawdown limit if you want the smallest
drops. Two half-strict rules are worse than one firm one.

**Step 3 — Pick your check-in rhythm.** With written rules, checking often is
free — daily reviews barely increase swapping, because the rules are the
brake, not the calendar; every 2–3 weeks was the weakest zone. **With written
rules, checking often adds safety without churn; with performance-chasing,
checking often adds churn without safety.** Every 1–5 trading days is a
sensible habit. (Review speed is taste, not edge — an earlier finding that
3-day reviews specifically helped did not survive the longer test window.)

**Step 4 — Choose your starting configuration and try to beat it.**
(1) *The honest default:* equal weight across as wide a pool as you can
operate, diversification cap on — near-zero effort, the champion to beat.
(2) *The cautious concentrator:* the multi-rule setup (streak + drawdown
limits, correlation cap) — the best return-per-worst-drop tested, a handful
of decisions a month. (3) *The simple concentrator:* the single 1,000-dollar
streak-cost rule. Build any of them on the **Build a Run** page in two
minutes — then spend your energy trying to beat the default, not tweaking
the winner.
""")

    st.subheader('Mini glossary')
    st.markdown("""
- **Drawdown (DD)** — how far the account fell from its best point. The pain number.
- **Sharpe ratio** — profit per unit of day-to-day wobble. Smoothness score.
- **Correlation** — do two robots win and lose *at the same time*? High correlation
  = one bad day hits both = hidden concentration.
- **Turnover** — how much swapping a style does. Churn has costs.
- **Review** — a scheduled check-in day where the rules are evaluated.
- **Lookback** — how far back the style looks when judging "recent form".
""")


# ══════════════════════════════════════════════════════════════════════════════
# 🗂 DATA
# ══════════════════════════════════════════════════════════════════════════════

elif page == '🗂 Data':
    st.title('🗂 Data management')
    st.caption('A **timeline** is a compiled bundle of backtest reports — the '
               'dataset every simulation reads. Build one from a folder of MT5 '
               'Strategy Tester `.htm` reports, or update an existing one after '
               'adding new reports.')

    # ── Existing timelines ────────────────────────────────────────────────
    st.subheader('Existing timelines')
    tl_rows = []
    for name in list_timelines():
        mpath = os.path.join(ENGINE_DIR, 'timeline', name, 'manifest.json')
        row = {'Timeline': name, 'Description': read_desc(name) or '—'}
        if os.path.isfile(mpath):
            with open(mpath, encoding='utf-8') as f:
                d = json.load(f).get('dataset', {})
            row.update({'Robots': d.get('ea_count'), 'Trades': d.get('trade_count'),
                        'From': d.get('first_trade'), 'To': d.get('last_trade'),
                        'Compiled': d.get('generated')})
        tl_rows.append(row)
    if tl_rows:
        st.dataframe(pd.DataFrame(tl_rows), use_container_width=True, hide_index=True)
    else:
        st.info('No timelines yet — compile one below.')

    # ── Describe / delete ─────────────────────────────────────────────────
    if tl_rows:
        st.subheader('Manage a timeline')
        mng = st.selectbox('Timeline to manage', list_timelines(),
                           help='Pick a timeline, then edit its description or '
                                'delete it below.')

        new_desc = st.text_area(
            'Description', value=read_desc(mng), height=80, key=f'desc_{mng}',
            help='A plain-language note about what this dataset is — e.g. which '
                 'reports went in, the date range, tick quality. Shown in the '
                 'table above and under the dataset selector on Build a Run.')
        if st.button('💾 Save description'):
            write_desc(mng, new_desc)
            st.success(f'Description saved for **{mng}**.')
            st.rerun()

        with st.expander('🗑 Delete this timeline'):
            st.markdown(f'This permanently removes the compiled dataset '
                        f'**{mng}** (`timeline/{mng}/`). The original `.htm` '
                        f'reports are **not** touched, so you can always '
                        f'recompile. Saved runs that used it stay in `runs/` '
                        f'but can no longer be re-run against it.')
            sure = st.checkbox(f'Yes, I want to delete **{mng}**',
                               key=f'del_confirm_{mng}')
            if st.button('🗑 Delete permanently', type='primary', disabled=not sure):
                shutil.rmtree(os.path.join(ENGINE_DIR, 'timeline', mng))
                cached_timeline.clear()
                cached_tradebook.clear()
                if st.session_state.get('dataset') == mng:
                    del st.session_state['dataset']
                st.success(f'Timeline **{mng}** deleted.')
                st.rerun()

    # ── Packaged suites ───────────────────────────────────────────────────
    st.subheader('📦 Packaged suites')
    suites = load_suites()
    st.caption(f'Defined in `{SUITES_PATH}` (shared with MT5Tools). A suite = '
               'one packaged-EA configuration (product × risk level) expressed '
               'as its list of individual main-pool strategies. The separate '
               'packaged backtests exist to (a) confirm which strategies each '
               'risk level runs and (b) validate below that grouping those '
               'strategies off the main pool matches the real package.')
    if not suites:
        st.info('No suites defined yet — see the `_readme` inside the file '
                'for the schema.')
    else:
        st.dataframe(pd.DataFrame([{
            'Suite': s['name'], 'Family': s.get('family', ''),
            'Strategies': len(s.get('members', [])) or 'not confirmed yet',
            'Package backtest (ea_id)': s.get('package_ea') or '—',
            'Notes': s.get('notes', '')} for s in suites]),
            use_container_width=True, hide_index=True)

        with st.expander('🔬 Validate a suite against its packaged backtest'):
            st.markdown('Compares the **sum of the suite\'s member strategies** '
                        '(from a main timeline) against the **standalone '
                        'packaged backtest** (from its own timeline), both on '
                        'the \\$100k linear basis. High daily correlation and '
                        'similar drawdown = the grouping is a faithful stand-in '
                        'and the package can be "hard-coded off main".')
            v_names = [s['name'] for s in suites if s.get('package_ea')]
            if not v_names:
                st.info('No suite has a standalone packaged backtest linked '
                        '(package_ea) — nothing to validate against.')
            v_suite = st.selectbox('Suite', v_names) if v_names else None
            sdef = (next(s for s in suites if s['name'] == v_suite)
                    if v_suite else {})
            tls = list_timelines()
            vc1, vc2 = st.columns(2)
            v_main = vc1.selectbox(
                'Main timeline (suite members)', tls,
                index=tls.index('main_pool_2018') if 'main_pool_2018' in tls
                      else 0)
            pk_default = sdef.get('package_timeline', '')
            v_pkg = vc2.selectbox(
                'Package timeline (standalone backtests)', tls,
                index=tls.index(pk_default) if pk_default in tls else 0)
            members = sdef.get('members', [])
            if not members:
                st.info('This suite has no member list yet — nothing to '
                        'validate.')
            else:
                m_daily, _ = cached_timeline(v_main)
                p_daily, p_meta = cached_timeline(v_pkg)
                have = [m for m in members if m in m_daily.columns]
                pk_opts = list(p_daily.columns)
                pk_ix = (pk_opts.index(sdef['package_ea'])
                         if sdef.get('package_ea') in pk_opts else 0)
                v_ea = st.selectbox('Packaged backtest to compare against',
                                    pk_opts, index=pk_ix,
                                    format_func=lambda e: friendly_name(e, p_meta))
                if len(have) < len(members):
                    st.warning(f'{len(members) - len(have)} member(s) missing '
                               f'from {v_main} — comparing the {len(have)} '
                               'present.')
                if have and st.button('Compare'):
                    g = m_daily[have].sum(axis=1)
                    p = p_daily[v_ea]
                    idx = g.index.intersection(p.index)
                    g, p = g.loc[idx], p.loc[idx]
                    corr = g.corr(p)
                    def _dd(x):
                        cum = x.cumsum() + 100_000
                        return float((cum - cum.cummax()).min())
                    stats = pd.DataFrame({
                        'Grouped off main': [g.sum(), _dd(g)],
                        'Packaged backtest': [p.sum(), _dd(p)],
                    }, index=['Total P&L ($)', 'Max drawdown ($)']).round(0)
                    st.metric('Daily P&L correlation', f'{corr:.3f}',
                              help='1.0 = the grouped strategies move '
                                   'identically to the package. Above ~0.95 '
                                   'the grouping is a faithful stand-in.')
                    st.dataframe(stats, use_container_width=True)
                    eq = pd.DataFrame({
                        'equity': g.cumsum() + 100_000}, index=idx)
                    ep = pd.DataFrame({
                        'equity': p.cumsum() + 100_000}, index=idx)
                    st.plotly_chart(equity_chart(
                        {'Grouped off main': eq, 'Packaged backtest': ep}),
                        use_container_width=True)
                    if corr >= 0.95:
                        st.success('The grouping tracks the package closely — '
                                   'safe to hard-code this suite off the main '
                                   'pool.')
                    else:
                        st.warning('Meaningful discrepancy — the grouping is '
                                   'directional, not a replication. Check the '
                                   'member list first (missing/extra '
                                   'strategies?); if that\'s right, the '
                                   'package itself differs structurally — '
                                   'e.g. a TradeFrequency throttle or one '
                                   'shared risk budget across the book (see '
                                   'the suite\'s notes). For package-level '
                                   'questions, prefer the package\'s own '
                                   'timeline.')

    # ── Compile / update ──────────────────────────────────────────────────
    st.subheader('Create or update a timeline')
    st.markdown("""
1. Copy the Strategy Tester `.htm` reports you want into a folder
   (subfolders are fine — the folder names become each robot's *family*).
2. Point the compiler at that folder, give the timeline a name, press Compile.

Using an **existing name updates that timeline in place** (all its future runs
use the new data); a **new name creates a separate dataset**, so you can keep
e.g. a gold-only pool and a full pool side by side.
""")
    folder = st.text_input('Reports folder',
                           value=os.path.join(ENGINE_DIR, 'reports_in'),
                           help='Scanned recursively for .htm reports. Duplicate '
                                'filenames are de-duplicated automatically.')
    new_name = st.text_input('Timeline name', value='main_pool',
                             help='Letters, numbers and underscores work best.')
    if st.button('⚙️ Compile timeline', type='primary',
                 disabled=not (folder.strip() and new_name.strip())):
        import io as _io
        import contextlib
        from compile_timeline import compile_reports
        if not os.path.isdir(folder):
            st.error(f'Folder not found: {folder}')
        elif not (glob.glob(os.path.join(folder, '**', '*.htm'), recursive=True)
                  or glob.glob(os.path.join(folder, '**', '*.html'),
                               recursive=True)):
            st.error('No .htm/.html reports found in that folder.')
        else:
            buf = _io.StringIO()
            out_dir = os.path.join(ENGINE_DIR, 'timeline', new_name.strip())
            try:
                with st.spinner('Parsing reports and building the timeline…'):
                    with contextlib.redirect_stdout(buf):
                        compile_reports(folder, out_dir)
                cached_timeline.clear()
                st.success(f'Timeline **{new_name.strip()}** compiled. '
                           'It is now available in the sidebar selector.')
            except SystemExit:
                st.error('Compile failed — details in the log below.')
            with st.expander('Compiler log', expanded=True):
                st.code(buf.getvalue() or '(no output)')

    # ── Market regime data ────────────────────────────────────────────────
    st.subheader('Market regime data')
    ref_path = os.path.join(ENGINE_DIR, 'reference', 'reference_prices.csv')
    if os.path.isfile(ref_path):
        _ref = pd.read_csv(ref_path, index_col='date', parse_dates=['date'])
        st.caption(f'Reference data: **{len(_ref.columns)} series**, latest date '
                   f'**{_ref.index.max():%d %b %Y}** '
                   f'({len(_ref)} rows, via Yahoo Finance).')
    else:
        st.caption('No reference data downloaded yet.')
    st.markdown('Refreshing downloads the latest reference prices (dollar index, '
                'VIX, etc.) and rebuilds the regime matrix for **every** timeline, '
                'so the Regimes page stays current.')
    if st.button('🔄 Refresh market data & rebuild regime matrices'):
        import subprocess
        py = os.path.join(ENGINE_DIR, '.venv', 'Scripts', 'python.exe')
        if not os.path.isfile(py):
            py = sys.executable
        logs = []
        ok = True
        with st.spinner('Downloading reference prices (Yahoo Finance)…'):
            r = subprocess.run([py, 'fetch_reference_data.py'], cwd=ENGINE_DIR,
                               capture_output=True, text=True, timeout=600)
            logs.append('── fetch_reference_data ──\n' + r.stdout + r.stderr)
            ok = ok and r.returncode == 0
        if ok:
            for tl in list_timelines():
                with st.spinner(f'Rebuilding regime matrix for {tl}…'):
                    r = subprocess.run([py, 'build_regime_matrix.py',
                                        '--timeline', tl], cwd=ENGINE_DIR,
                                       capture_output=True, text=True, timeout=300)
                    logs.append(f'── build_regime_matrix {tl} ──\n' + r.stdout + r.stderr)
                    ok = ok and r.returncode == 0
        if ok:
            st.success('Market data refreshed and regime matrices rebuilt for all timelines.')
        else:
            st.error('Refresh hit an error — see the log below. If Yahoo failed, '
                     'the Dukascopy fallback is documented in fetch_reference_data.py.')
        with st.expander('Refresh log', expanded=not ok):
            st.code('\n\n'.join(logs))

    with st.expander('Where do reports come from?'):
        st.markdown("""
Reports are produced by the **MT5 Tools batch backtester** (or any MT5 Strategy
Tester run saved as an `.htm` report). For this project's conventions — fixed
100k balance, lot step calibrated to the 5% historical-DD target — see the
README in the engine folder. The compiler reads each report's trades, its
input parameters (lot step, historical max DD) and its results summary, and
flags robots whose drawdown calibration looks stale or defaulted.
""")


# ══════════════════════════════════════════════════════════════════════════════
# 📊 EA POOL
# ══════════════════════════════════════════════════════════════════════════════

elif page == '📊 EA Pool':
    st.title('📊 The robot pool')
    if not timeline_name:
        st.stop()
    daily, meta = cached_timeline(timeline_name)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Robots (EAs)', len(meta))
    c2.metric('Strategy families', meta.family.nunique())
    c3.metric('Markets', meta.symbol.nunique())
    c4.metric('Trading days', len(daily))

    st.caption('Every row is one robot: a strategy configuration backtested '
               'on a fixed $100k account (the main pool covers 2018 → 2026).')

    fam = st.multiselect('Filter by family (source folder)',
                         sorted(meta.family.unique()))
    sym = st.multiselect('Filter by market', sorted(meta.symbol.unique()))
    view = meta.copy()
    if fam:
        view = view[view.family.isin(fam)]
    if sym:
        view = view[view.symbol.isin(sym)]

    show = view[['strategy', 'family', 'symbol', 'timeframe', 'net_profit',
                 'realized_dd_pct', 'dd_vs_target', 'trades']].rename(columns={
        'strategy': 'Strategy', 'family': 'Family', 'symbol': 'Market',
        'timeframe': 'TF', 'net_profit': 'Backtest profit ($)',
        'realized_dd_pct': 'Worst DD (%)', 'dd_vs_target': 'DD vs 5% target',
        'trades': 'Trades'})
    st.dataframe(show, use_container_width=True, hide_index=True,
                 column_config={
                     'DD vs 5% target': st.column_config.NumberColumn(
                         help='1.0 = this robot\'s backtest drawdown landed exactly on '
                              'the 5% calibration target. Far below 1 = it risked less '
                              'than intended here; far above = more. Values far from 1 '
                              'usually mean the set author\'s historical-DD number was '
                              'stale or a default.')})

    with st.expander('What does "DD vs 5% target" mean, and why care?'):
        st.markdown("""
Each robot's trade size was set using a **historical max drawdown** number provided
in its settings file. If that number was accurate, the backtest's worst drawdown
should land near **5% of the account** (ratio ≈ 1.0). A ratio of 0.2 means the robot
risked far *less* than intended; 3.0 means far *more*. Robots built by third parties
sometimes ship with stale or default numbers — this column is the lie detector.
**History is not a limit either way**: even a perfectly calibrated robot can exceed
its historical worst in the future.
""")


# ══════════════════════════════════════════════════════════════════════════════
# 🌐 REGIMES
# ══════════════════════════════════════════════════════════════════════════════

elif page == '🌐 Regimes':
    st.title('🌐 Market regimes — where does each robot make its money?')
    if not timeline_name:
        st.stop()
    daily, meta = cached_timeline(timeline_name)
    tdir = os.path.join(ENGINE_DIR, 'timeline', timeline_name)
    states_path = os.path.join(tdir, 'regime_states.csv')
    matrix_path = os.path.join(tdir, 'regime_matrix.csv')

    if not (os.path.isfile(states_path) and os.path.isfile(matrix_path)):
        st.info('No regime data for this timeline yet. From the engine folder run:\n\n'
                '```\npython fetch_reference_data.py\n'
                f'python build_regime_matrix.py --timeline {timeline_name}\n```')
        st.stop()

    states = pd.read_csv(states_path, index_col='date', parse_dates=['date'])
    matrix = pd.read_csv(matrix_path)

    st.markdown("""
Markets move through **regimes** — stretches where the dollar is strong or weak,
volatility is calm or stressed, Bitcoin is in a bull or bear phase. A robot that
looks brilliant overall may earn *all* of its money in one regime and tread
water (or bleed) in the others. This page shows **where each robot's profit
historically came from**, so a team isn't accidentally a one-regime bet —
which is exactly how the "team of 9 Bitcoin robots" trap happens.
""")
    with st.expander('⚠️ How to use this honestly (please read once)'):
        st.markdown("""
- **This is a description of the past, not a prediction.** The states are
  *coincident*: "gold robots earned most in gold uptrends" does not tell you
  when the next uptrend starts or ends.
- States come from simple transparent rules — price vs its own 100/200-day
  average, and fixed VIX levels (calm <15, normal 15–25, stressed >25). No
  fitted models, nothing clever hiding inside.
- Sensible uses: check a candidate team isn't concentrated in one regime;
  understand *why* a robot is slumping ("its regime is out"); set expectations.
  Risky use: flipping robots on/off the moment a state changes — test that as
  a regime in Build a Run before believing it.
""")

    # ── Current states ────────────────────────────────────────────────────
    st.subheader('Where the market is right now')
    latest = states.dropna().iloc[-1]
    st.caption(f'As of {states.dropna().index[-1]:%d %b %Y} (data via Yahoo Finance):')
    cols = st.columns(len(latest))
    for c, (ind, val) in zip(cols, latest.items()):
        c.metric(ind, str(val))

    # ── Heatmap ───────────────────────────────────────────────────────────
    st.subheader('Profit smoothness by regime (Sharpe)')
    level = st.radio('Show', ['Families + whole pool', 'Individual robots'],
                     horizontal=True)
    if level == 'Individual robots':
        f1, f2 = st.columns(2)
        fams = f1.multiselect('Filter families', sorted(meta.family.unique()))
        mkts = f2.multiselect('Filter markets', sorted(meta.symbol.unique()),
                              help='e.g. pick XAUUSD.a to see every gold robot '
                                   'across all families side by side.')
        keep = meta
        if fams:
            keep = keep[keep.family.isin(fams)]
        if mkts:
            keep = keep[keep.symbol.isin(mkts)]
        sub = matrix[(matrix.type == 'ea') & matrix.entity.isin(keep.ea_id.tolist())]
    else:
        sub = matrix[matrix.type.isin(['family', 'pool', 'suite'])]
        # Show the packaged-EA configurations alongside the pool's families —
        # each package is one risk unit, same footing as one family robot.
        if timeline_name != 'packaged_suites':
            pk_path = os.path.join(ENGINE_DIR, 'timeline', 'packaged_suites',
                                   'regime_matrix.csv')
            if os.path.isfile(pk_path):
                pk = pd.read_csv(pk_path)
                pk = pk[pk.type == 'ea'].copy()
                pk['entity'] = '📦 ' + pk['entity']
                sub = pd.concat([sub, pk], ignore_index=True)
                st.caption('📦 rows are the packaged-EA configurations (from '
                           'the packaged_suites dataset, one risk unit each) '
                           'shown for comparison — their day count is shorter '
                           'than the pool\'s, so compare Sharpe, not totals.')

    sub = sub.copy()
    sub['col'] = sub['indicator'] + ': ' + sub['state']
    heat = sub.pivot_table(index='entity', columns='col', values='sharpe')
    if len(heat) > 0:
        # Absolute anchors at the bottom, relative at the top: red = Sharpe 0
        # or below (no edge), amber at ~1 (where a real edge conventionally
        # begins), green deepening from there to the strongest value shown.
        EDGE_SHARPE = 1.0
        finite = heat.values[np.isfinite(heat.values)]
        zmin = float(min(0.0, np.percentile(finite, 5))) if finite.size else 0.0
        zmax = (float(max(np.percentile(finite, 95), EDGE_SHARPE * 2))
                if finite.size else EDGE_SHARPE * 2)
        stops = [(0.0, '#d73027')]
        if zmin < 0:
            stops.append(((0.0 - zmin) / (zmax - zmin), '#d73027'))
        stops.append(((EDGE_SHARPE - zmin) / (zmax - zmin), '#fee08b'))
        stops.append((1.0, '#1a9850'))
        fig = go.Figure(go.Heatmap(
            z=heat.values, x=list(heat.columns), y=list(heat.index),
            colorscale=stops, zmin=zmin, zmax=zmax,
            colorbar=dict(title='Sharpe'),
            hovertemplate='%{y}<br>%{x}<br>Sharpe %{z:.2f}<extra></extra>'))
        fig.update_layout(height=max(300, 26 * len(heat) + 120),
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(tickangle=-35))
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Color anchors: red = Sharpe 0 or below (no edge), amber '
                   '≈ Sharpe 1 — the conventional line where a real edge '
                   'begins — and green deepens from there to the strongest '
                   'value in this view. A robot that is green in one column '
                   'and red in its opposite is a one-regime specialist — '
                   'fine, as long as the team knows it and balances around '
                   'it.')

    # ── Drill-down ────────────────────────────────────────────────────────
    st.subheader('Drill into a family — or specific strategies within it')
    st.caption('A family row includes **every** robot from that folder, equally '
               'weighted. Use the second box to narrow to specific strategies — '
               'one strategy shows its own numbers; several show the combined '
               'result of just that subset (recomputed live).')
    c1, c2 = st.columns(2)
    fam_opts = ['POOL: all robots'] + ['FAMILY: ' + f for f in sorted(meta.family.unique())]
    scope = c1.selectbox('Family / whole pool', fam_opts)
    members = []
    if scope.startswith('FAMILY: '):
        fam_meta = meta[meta.family == scope[8:]]
        members = c2.multiselect(
            'Strategies within family (blank = whole family)',
            fam_meta.ea_id.tolist(),
            format_func=lambda e: fam_meta.set_index('ea_id').at[e, 'strategy']
                                  + f" ({fam_meta.set_index('ea_id').at[e, 'symbol']})")

    if len(members) == 1:
        d = matrix[matrix.entity == members[0]].copy()
        st.caption(f'Showing **{members[0]}** on its own.')
    elif len(members) > 1:
        from build_regime_matrix import condition as _condition
        subset_pnl = daily[[m for m in members if m in daily.columns]].sum(axis=1)
        recs = []
        for ind in states.columns:
            for row in _condition(subset_pnl, states[ind]):
                recs.append({'indicator': ind, **row})
        d = pd.DataFrame(recs)
        st.caption(f'Showing the combined result of **{len(members)} selected '
                   'strategies** (equal weight, recomputed live).')
    else:
        d = matrix[matrix.entity == scope].copy()
    d['col'] = d['indicator'] + ': ' + d['state']
    show = d[['indicator', 'state', 'days', 'total_pnl', 'avg_daily',
              'sharpe', 'pnl_share_pct']].rename(columns={
        'indicator': 'Indicator', 'state': 'Regime', 'days': 'Days',
        'total_pnl': 'Profit ($)', 'avg_daily': 'Avg per day ($)',
        'sharpe': 'Sharpe', 'pnl_share_pct': 'Share of profit (%)'})
    st.dataframe(show, use_container_width=True, hide_index=True)
    fig = go.Figure(go.Bar(x=d['col'], y=d['sharpe'],
                           marker_color=np.where(d['sharpe'] >= 0,
                                                 '#2dc653', '#e05555')))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title='Sharpe', xaxis=dict(tickangle=-35))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🛠 BUILD A RUN
# ══════════════════════════════════════════════════════════════════════════════

elif page == '🛠 Build a Run':
    st.title('🛠 Build a management style and test it')
    timeline_name = dataset_selector('dataset_build_run', show_desc=True)
    if not timeline_name:
        st.stop()
    daily, meta = cached_timeline(timeline_name)
    ea_ids = list(daily.columns)

    st.caption('Configure a style below, then press Run. The simulator walks '
               'history day by day — decisions only ever use information that '
               'was available at the time.')

    run_name = st.text_input('Name this run', placeholder='e.g. my_first_rules_test',
                             help='Results are saved under runs/<name> so you can '
                                  'compare styles later.')

    REGIME_INFO = {
        'equal_weight': ('Equal weight — everyone plays',
                         'Every selected robot runs at the same size, forever. '
                         'No decisions. The benchmark to beat.'),
        'static_topn' : ('Static top-N — pick once, never touch',
                         'Ranks robots on early history, fields the top N, then '
                         'never changes the team again.'),
        'momentum'    : ('Momentum — chase recent form',
                         'At every review, re-ranks everyone by recent performance '
                         'and fields the current top N. High churn.'),
        'rules'       : ('Rules — bench on damage, promote the best sub',
                         'Only benches a robot when it breaks one of your written '
                         'rules (losing streak / drawdown), then promotes the best '
                         'available substitute. Low churn.'),
        'inverse_vol' : ('Inverse volatility — size down the wild ones',
                         'Everyone plays, but calmer robots get more size and '
                         'wilder robots get less.'),
        'random'      : ('Random — dice-roll swaps (control test)',
                         'Swaps robots randomly. Only useful to prove the other '
                         'styles beat luck.'),
    }
    regime = st.selectbox('Management style', list(REGIME_INFO),
                          format_func=lambda k: REGIME_INFO[k][0])
    st.info(REGIME_INFO[regime][1])

    # ── Test period & risk size ───────────────────────────────────────────
    st.subheader('1 — Test period & robot size')
    dmin, dmax = daily.index[0].date(), daily.index[-1].date()
    c1, c2, c3 = st.columns(3)
    start_date = c1.date_input('Start date', dmin, min_value=dmin, max_value=dmax,
                               help='Start later than the data begins to test on a '
                                    'period the robots were NOT tuned on. If these '
                                    'set files were optimised on pre-2023 data, '
                                    'starting in 2023 gives a more honest test '
                                    'and helps avoid overfitting.')
    end_date   = c2.date_input('End date', dmax, min_value=dmin, max_value=dmax,
                               help='Stop early to hold back recent data for a '
                                    'later out-of-sample check.')
    risk_pct   = c3.slider('Account MaxDD % risk applied using lot steps',
                           0.5, 50.0, 5.0, 0.5,
                           help='Lot sizes are set by the lot-step method: each '
                                'robot\'s lots are calibrated so its HISTORICAL '
                                'max drawdown equals this % of the account. '
                                '5% is a comfortable median for someone getting '
                                'into this. Higher settings (30–50%) are '
                                'aggressive configurations typically run on a '
                                'small satellite account holding only 10–15% of '
                                'total capital — risk diversified ACROSS '
                                'accounts, not just within one. Sizing is '
                                'linear: 10% runs every robot at exactly double '
                                'the 5% baseline. History is not a limit either '
                                'way.')
    _scale = risk_pct / 5.0
    st.caption(f'At **{risk_pct:g}%** per robot, the Monte Carlo expectations for '
               f'a 10-slot rules book scale to: normal worst drop '
               f'**~{5.3 * _scale:.0f}%** of the account, unlucky (95th pct) '
               f'**~{9 * _scale:.0f}%**. '
               + ('⚠️ At this level a normal bad stretch is a major drawdown — '
                  'appropriate only for a small slice of total capital, with '
                  'the "broken vs normal" thresholds written down in advance.'
                  if risk_pct > 15 else
                  'Decide in advance which number means "broken".'))
    if start_date >= end_date:
        st.error('Start date must be before end date.')
        st.stop()

    # ── Team selection ────────────────────────────────────────────────────
    st.subheader('2 — Pick the team')
    pick_mode = st.radio('Starting team',
                         ['Auto-pick the top N', 'Auto-pick for the regime',
                          'Pick the team yourself'],
                         horizontal=True,
                         help='Auto-pick (top N) ranks robots by early-history '
                              'Sharpe (first ~3 months). Auto-pick (regime) '
                              'reads the market regime at your start date — '
                              'dollar trend, stocks, gold, VIX etc. — and '
                              'fields the N robots with the best form in that '
                              'regime, judged only on history before the start. '
                              'Pick yourself gives you filters (family, market, '
                              'timeframe) plus a hand-pick list.')
    n_slots = 10
    portfolio_spec = 'top10_sharpe'
    candidate_pool = None
    if pick_mode == 'Auto-pick the top N':
        af1, af2 = st.columns(2)
        ap_fams = af1.multiselect(
            'Family filter (auto-pick pool)', sorted(meta.family.unique()),
            help='Auto-pick ranks only robots from these families — e.g. '
                 'pick the gold families for "the top N gold robots". Empty '
                 '= the whole pool (careful: unfiltered early-Sharpe ranking '
                 'historically picks ~9 Bitcoin robots — the concentration '
                 'trap).')
        ap_syms = af2.multiselect(
            'Market filter (auto-pick pool)', sorted(meta.symbol.unique()),
            help='Only robots trading these markets. Empty = all markets.')
        n_slots = st.number_input('Team size (N slots)', 2, 40, 10,
                                  help='How many robots trade at once. Each filled '
                                       'slot carries one risk unit (~5% historical '
                                       'DD).')
        portfolio_spec = f'top{int(n_slots)}_sharpe'
        if ap_fams or ap_syms:
            ap_meta = meta
            if ap_fams:
                ap_meta = ap_meta[ap_meta.family.isin(ap_fams)]
            if ap_syms:
                ap_meta = ap_meta[ap_meta.symbol.isin(ap_syms)]
            candidate_pool = ap_meta.ea_id.tolist()
            if len(candidate_pool) < int(n_slots):
                st.warning(f'Only {len(candidate_pool)} robot(s) match the '
                           'filter — fewer than the team size.')
            else:
                st.caption(f'🎯 Auto-pick will choose the top {int(n_slots)} '
                           f'from the {len(candidate_pool)} robot(s) matching '
                           'the filter.')
    if pick_mode == 'Auto-pick for the regime':
        portfolio_spec = []
        reg_path = os.path.join(ENGINE_DIR, 'timeline', timeline_name,
                                'regime_states.csv')
        if not os.path.isfile(reg_path):
            st.info('No regime data for this timeline yet. From the engine '
                    'folder run:\n\n```\npython build_regime_matrix.py '
                    f'--timeline {timeline_name}\n```')
        else:
            states = pd.read_csv(reg_path, parse_dates=['date']).set_index('date')
            start_ts = pd.Timestamp(start_date)
            if len(daily[daily.index < start_ts]) < 90 and len(daily) > 90:
                # The pick judges form on history BEFORE the start, so an
                # early start has nothing to judge with — default ~4 months
                # in and ask, rather than just warning.
                auto_start = daily.index[90].date()
                start_date = st.date_input(
                    'Start date for this run', auto_start,
                    min_value=auto_start, max_value=daily.index[-1].date(),
                    key='regime_pick_start',
                    help='The regime pick ranks robots on history BEFORE the '
                         'start date, so the run begins ~4 months into the '
                         'dataset at the earliest. This overrides the start '
                         'date in section 1 for this run.')
                start_ts = pd.Timestamp(start_date)
                st.caption(f'📅 Run starts **{start_ts:%d %b %Y}** (the first '
                           '~4 months stay reserved as pre-start evidence).')
            hist = daily[daily.index < start_ts]
            known = states[states.index < start_ts]
            if len(hist) < 90 or known.empty:
                st.warning('Not enough history before your start date to judge '
                           'regime form — move the start date at least ~4 '
                           'months into the dataset.')
            else:
                now_state = known.iloc[-1]
                n_slots = int(st.number_input(
                    'Team size (N slots)', 2, 40, 10,
                    help='How many robots trade at once. Each filled slot '
                         'carries one risk unit (~5% historical DD).'))
                cond = st.selectbox(
                    'Judge form against which regime?',
                    ['All markets combined'] + list(states.columns),
                    help='"All markets combined" scores each robot in every '
                         'current regime state (dollar, stocks, gold, crypto, '
                         'rates, oil, VIX) and averages. Or condition on a '
                         'single market\'s state — e.g. only "how does it do '
                         'while gold is in an uptrend".')
                st_hist = states.reindex(hist.index, method='ffill')
                inds = (list(states.columns) if cond == 'All markets combined'
                        else [cond])
                score = pd.DataFrame(index=hist.columns)
                slice_txt, thin = [], []
                for i in inds:
                    mask = (st_hist[i] == now_state[i]).values
                    sub = hist[mask]
                    slice_txt.append(f'{now_state[i]} · {int(mask.sum())}d')
                    if mask.sum() < 40:
                        thin.append(str(now_state[i]))
                    mu, sd = sub.mean(), sub.std()
                    score[i] = np.where(sd > 0, mu / sd * np.sqrt(252), np.nan)
                ranked = (score.mean(axis=1, skipna=True)
                          .sort_values(ascending=False).dropna())
                chosen = ranked.head(n_slots).index.tolist()
                portfolio_spec = chosen
                st.caption(f'Regime at your start date ({start_ts:%d %b %Y}): '
                           f'**{" — ".join(slice_txt)}** (state · matching '
                           'pre-start trading days). Robots are ranked by '
                           'Sharpe on those days only.')
                if thin:
                    st.warning('Thin evidence: fewer than 40 pre-start days '
                               'match ' + ', '.join(f'**{t}**' for t in thin) +
                               ' — form judged on so few days is mostly noise.')
                with st.expander(f'The {len(chosen)} robots the regime pick '
                                 'chose', expanded=False):
                    for e in chosen:
                        st.markdown(f'- {friendly_name(e, meta)} — regime '
                                    f'Sharpe {ranked[e]:.1f}')
                st.info('**Honest note:** this is *descriptive*, not '
                        'predictive — it fields the robots that historically '
                        'did well in conditions like the start date\'s, but '
                        'regimes flip without warning and the tool has no way '
                        'of knowing when. Compare it against a plain top-N '
                        'pick before trusting it.')

    elif pick_mode == 'Pick the team yourself':
        suites = load_suites()
        suite_default = []
        if suites:
            suite_sel = st.selectbox(
                'Suite / portfolio quick-pick', ['(no suite)'] +
                [s['name'] for s in suites],
                help='Defined in packaged_suites.json — packaged-EA '
                     'configurations (product × risk level) and curated '
                     'portfolios like Wim\'s. Picking one pre-selects those '
                     'robots below.')
            if suite_sel != '(no suite)':
                s = next(s for s in suites if s['name'] == suite_sel)
                members = s.get('members', [])
                if not members:
                    st.info('This suite has no strategy list yet — fill in '
                            'its "members" in packaged_suites.json once the '
                            'packaged backtest confirms which strategies the '
                            'risk level runs.')
                suite_default = [m for m in members if m in ea_ids]
                missing = [m for m in members if m not in ea_ids]
                if missing:
                    st.warning(f'{len(missing)} suite member(s) not in this '
                               'dataset: ' + ', '.join(missing[:5]) +
                               ('…' if len(missing) > 5 else ''))
        f1, f2, f3 = st.columns(3)
        fam_pick = f1.multiselect('Strategy family', sorted(meta.family.unique()),
                                  help='Robots whose reports came from these '
                                       'folders. Pick several to combine suites — '
                                       'e.g. Gold Reaper + Goldtrade Pro. Empty '
                                       '= all families.')
        sym_pick = f2.multiselect('Market (symbol)', sorted(meta.symbol.unique()),
                                  help='Only robots trading these markets. '
                                       'Empty = all markets.')
        tf_pick  = f3.multiselect('Timeframe', sorted(meta.timeframe.unique()),
                                  help='Only robots running on these chart '
                                       'timeframes. Empty = all timeframes.')
        fmeta = meta
        if fam_pick:
            fmeta = fmeta[fmeta.family.isin(fam_pick)]
        if sym_pick:
            fmeta = fmeta[fmeta.symbol.isin(sym_pick)]
        if tf_pick:
            fmeta = fmeta[fmeta.timeframe.isin(tf_pick)]
        pool = fmeta.ea_id.tolist()
        filtered = bool(fam_pick or sym_pick or tf_pick)
        suite_default = [m for m in suite_default if m in pool]
        chosen = st.multiselect(
            'Team robots', pool,
            default=suite_default or (pool if filtered else []),
            format_func=lambda e: friendly_name(e, meta),
            help='With a filter on, every match starts selected — deselect any '
                 'you don\'t want. With no filter, hand-pick from the whole '
                 'pool. (Changing a filter resets this list to the new '
                 'matches.)')
        if chosen:
            per_fam = ' · '.join(f"{f}: {n}" for f, n in
                                 meta[meta.ea_id.isin(chosen)]
                                 .family.value_counts().items())
            st.caption(f'**{len(chosen)} robot(s)** on the team — {per_fam}')
        portfolio_spec = chosen

    if pick_mode == 'Pick the team yourself' and chosen:
        if regime in ('equal_weight', 'inverse_vol'):
            n_slots = len(chosen)
            st.number_input('Team size (N slots)', value=len(chosen), disabled=True,
                            help='With this management style every picked robot '
                                 'plays, so team size is simply how many you '
                                 'picked.')
        elif len(chosen) < 3:
            n_slots = len(chosen)
        else:
            n_slots = int(st.number_input(
                'Team size (N slots)', 2, len(chosen), len(chosen),
                help='How many of your picked robots are fielded at once. '
                     'Leave it at the maximum to field everyone. Lower it and '
                     'the style chooses within your pool: top-N / momentum / '
                     'random rank or draw from your picks, rules starts with '
                     'the first N you selected (the rest can still come in as '
                     'substitutes).'))
            if n_slots < len(chosen):
                st.caption(f'⚖️ Fielding **{n_slots} of {len(chosen)}** picked '
                           'robots — the management style decides which.')

    capacity, fill_after = int(n_slots), 0
    if regime == 'rules':
        cc1, cc2 = st.columns(2)
        capacity = int(cc1.number_input(
            'Team capacity (slots)', int(n_slots), 40, int(n_slots),
            help='Your upper limit of simultaneous robots. Set it higher '
                 'than the starting team to keep BLANK slots in reserve — '
                 'e.g. capacity for 15 while starting with 10.'))
        if capacity > int(n_slots):
            fill_after = int(cc2.number_input(
                'Fill blank slots after (trading days)', 0, 252, 63,
                help='The refill may only use the blank slots after this '
                     'many trading days — 63 ≈ 3 months, once enough '
                     'evidence has accumulated. Benched robots are still '
                     'replaced immediately (up to the starting team size).'))
            st.caption(f'⚖️ Starting with **{int(n_slots)} of {capacity} '
                       f'slots filled** — {capacity - int(n_slots)} blank '
                       f'slot(s) held in reserve, unlocked for the refill '
                       f'after {fill_after} trading days. Each slot is one '
                       'risk unit whether filled or blank, so deployed risk '
                       'steps up when the blanks fill.')

    subs_spec = 'all'
    if regime == 'rules':
        sub_mode = st.radio('Substitutes bench', ['All other robots', 'Choose manually'],
                            horizontal=True,
                            help='Who can be promoted when someone is benched. '
                                 'Note: your STARTING team is always eligible '
                                 'to return from the bench after its cooldown, '
                                 'even if not in this list — so a gold-only '
                                 'bench only keeps out gold-only NEWCOMERS; '
                                 'restrict the starting team too (auto-pick '
                                 'filter / pick yourself) for a fully '
                                 'gold-only book.')
        if sub_mode == 'Choose manually':
            sub_fams = st.multiselect(
                'Substitute family filter', sorted(meta.family.unique()),
                help='Narrow the bench to these families — e.g. only gold '
                     'robots may substitute into a gold book. Empty = all '
                     'families. (Changing the filter resets the robot list '
                     'to the new matches.)')
            spool = (meta[meta.family.isin(sub_fams)] if sub_fams
                     else meta).ea_id.tolist()
            subs_spec = st.multiselect(
                'Substitute robots', spool,
                default=spool if sub_fams else [],
                format_func=lambda e: friendly_name(e, meta),
                help='With a family filter on, every match starts selected — '
                     'deselect any you don\'t want. With no filter, '
                     'hand-pick from the whole pool.')

    # ── Rules / params ────────────────────────────────────────────────────
    st.subheader('3 — Set the rules')
    params = {}
    colA, colB = st.columns(2)
    with colA:
        review_every = st.slider('Review every N trading days', 1, 21, 5,
                                 help='How often the manager checks the rules. '
                                      '5 = weekly-ish. More frequent = more churn.')
        lookback = st.slider('Lookback window (trading days)', 21, 252, 63,
                             help='How much recent history counts as "recent form". '
                                  '63 ≈ 3 months.')
    with colB:
        warmup = st.slider('Warm-up before first decision (days)', 21, 252, 63,
                           help='The simulator stays flat this long so early '
                                'decisions have some history to work with.')
        metric = st.selectbox('Form metric', ['sharpe', 'return', 'calmar'],
                              help='How "recent form" is scored. Sharpe = smooth '
                                   'profit; return = raw profit; calmar = profit '
                                   'vs worst drop.')
    params['lookback'] = lookback
    params['metric']   = metric

    if regime == 'rules':
        st.markdown('**Benching rules** — a robot is benched when *any* enabled rule fires.')
        streak_mode = st.radio(
            'Count losing streaks in…', ['days', 'trades'], horizontal=True,
            help='**Days** = consecutive losing trading days (only days the robot '
                 'actually traded count). **Trades** = consecutive losing trades from '
                 'the trade history — fairer when comparing a fast robot that trades '
                 '20× a day against a slow one that trades once a day.')
        unit = streak_mode
        c1, c2 = st.columns(2)
        with c1:
            use_streak = st.checkbox('Bench on losing streak', True)
            streak = st.slider(f'…after this many losing {unit} in a row', 2, 15, 5,
                               disabled=not use_streak)
            use_dollar = st.checkbox('Bench on streak cost', False,
                                     help='Fires when the CURRENT losing streak has cost '
                                          'more than this many dollars in total — catches '
                                          'a short but brutal streak that a simple count '
                                          'would miss.')
            dollar_lim = st.number_input('…streak cost threshold ($)', 250, 20000, 3000,
                                         step=250, disabled=not use_dollar)
            use_freq = st.checkbox('Bench on loss frequency', False,
                                   help='NOT consecutive — fires when there are this many '
                                        f'losing {unit} inside a recent window, even with '
                                        'wins sprinkled in between. Catches the slow bleeder.')
            freq_n = st.slider('…this many losses', 3, 30, 8, disabled=not use_freq)
            freq_m = st.slider('…within this many trading days', 5, 63, 21,
                               disabled=not use_freq)
        with c2:
            use_dd = st.checkbox('Bench on drawdown', True)
            dd_lim = st.slider('…after it drops this % of the account', 0.5, 6.0, 2.5, 0.5,
                               disabled=not use_dd,
                               help='Measured over the lookback window, per robot, at '
                                    'full backtest size.')
            use_corr = st.checkbox('Correlation cap on promotions', False,
                                   help='Blocks promoting a substitute that wins/loses '
                                        'at the same time as the current team. Leave OFF '
                                        'for a deliberately concentrated team (e.g. all-gold).')
            corr_cap = st.slider('…max correlation allowed', 0.3, 0.95, 0.7, 0.05,
                                 disabled=not use_corr)
            cooldown = st.slider('Cooldown before a benched robot can return (days)',
                                 0, 63, 21)
            pick_top = st.slider('Promote randomly from the top K candidates',
                                 1, 10, 1,
                                 help='1 = always promote the single best '
                                      'available — deterministic, so every '
                                      'run gravitates to the same star '
                                      'robots. Above 1, each refill picks at '
                                      'random among the K best eligible, '
                                      'spreading promotions across '
                                      'near-equals. Reproducible: the same '
                                      'run gives the same picks.')
        params.update({
            'streak_mode'        : streak_mode,
            'loss_streak_limit'  : streak if use_streak else None,
            'streak_dollar_limit': int(dollar_lim) if use_dollar else None,
            'loss_count_limit'   : int(freq_n) if use_freq else None,
            'loss_count_window'  : int(freq_m),
            'ea_dd_limit_pct'    : dd_lim if use_dd else None,
            'corr_cap'           : corr_cap if use_corr else None,
            'cooldown_days'      : cooldown,
            'pick_from_top'      : int(pick_top),
        })

    if regime == 'momentum':
        use_mcost = st.checkbox(
            'Exclude robots on a costly losing streak', False,
            help='Momentum still picks the top N by recent form, but a robot '
                 'whose CURRENT losing streak has cost more than this threshold '
                 'is ineligible until the streak ends — stops promoting a robot '
                 'mid-bleed just because its longer history still ranks well.')
        mcost = st.number_input('…streak cost threshold ($)', 250, 20000, 1500,
                                step=250, disabled=not use_mcost)
        params['streak_dollar_limit'] = int(mcost) if use_mcost else None

    if regime in ('rules', 'momentum'):
        use_symcap = st.checkbox(
            'Limit robots per market (diversification)', False,
            help='Caps how many team slots one market can occupy — e.g. at most '
                 '3 gold robots at a time. This is the rule that stops "pick the '
                 'best performers" from quietly building a team of 9 Bitcoin '
                 'robots. Applies to the starting team AND every later promotion.')
        sym_cap = st.slider('…max robots on the same market', 1, 10, 3,
                            disabled=not use_symcap)
        params['max_per_symbol'] = int(sym_cap) if use_symcap else None

    # ── Overlays ──────────────────────────────────────────────────────────
    st.subheader('4 — Optional safety overlays')
    overlays = {}
    c1, c2 = st.columns(2)
    with c1:
        if st.checkbox('Drawdown de-risking', False,
                       help='Automatically shrinks the whole team\'s size as the '
                            'account falls from its peak, and restores it on recovery. '
                            'The disciplined version of "getting scared" — cuts pain, '
                            'can slow recovery.'):
            start = st.slider('Start shrinking at account DD %', 1.0, 8.0, 3.0, 0.5)
            floor = st.slider('Minimum size reached at DD %', start + 0.5, 12.0, 6.0, 0.5)
            overlays['dd_derisk'] = {'start_pct': start, 'floor_pct': floor}
    with c2:
        if st.checkbox('Volatility targeting', False,
                       help='Keeps the team\'s day-to-day wobble near a target by '
                            'scaling everyone up in quiet times and down in wild times.'):
            tv = st.number_input('Target yearly wobble ($)', 5_000, 60_000, 15_000,
                                 step=1_000)
            overlays['vol_target'] = {'target_ann_vol': tv}

    # ── Run ───────────────────────────────────────────────────────────────
    st.divider()
    _gross = float(capacity) * float(risk_pct) / 5.0
    st.caption(f'⚖️ This run **sums {int(capacity)} robot(s) at '
               f'{risk_pct:.1f}% sizing** — {_gross:.0f} robots-worth of '
               'backtested size in total ("risk units"). Thanks to the '
               '\\$100k / 5%-DD normalisation the backtest figures add '
               'directly, so profit and drawdown dollars grow with every '
               'robot you add — a 37-robot book shows ~3.7× the dollars of '
               'a 10-robot bench purely from summing more robots. Compare '
               'different-sized runs on Sharpe, or rerun at a matching '
               'count.')
    problems = []
    if not run_name.strip():
        problems.append('**give the run a name** (the box at the top of the page — '
                        'results are saved under it)')
    if pick_mode == 'Auto-pick for the regime' and not portfolio_spec:
        problems.append('**the regime pick has no team yet** — it needs regime '
                        'data for this dataset and enough history before the '
                        'start date (see the message in section 2)')
    elif pick_mode != 'Auto-pick the top N' and not portfolio_spec:
        problems.append('**pick at least one robot** for the team '
                        f'(the "{pick_mode}" box is empty)')
    if regime == 'rules' and isinstance(subs_spec, list) and not subs_spec:
        problems.append('**pick some substitutes**, or switch the bench back to '
                        '"All other robots"')
    if problems:
        st.warning('Before you can run:\n\n' + '\n'.join(f'- {p}' for p in problems))
    if st.button('▶ Run the simulation', type='primary',
                 disabled=bool(problems)):
        cfg = {'timeline': timeline_name, 'regime': regime,
               'portfolio': portfolio_spec, 'substitutes': subs_spec,
               'candidate_pool': candidate_pool,
               'n_slots': int(capacity),
               'capacity': int(capacity),
               'fill_blanks_after': int(fill_after),
               'gross_budget': float(capacity) * float(risk_pct) / 5.0,
               'risk_pct_per_ea': float(risk_pct),
               'start_date': str(start_date), 'end_date': str(end_date),
               'review_every': int(review_every), 'warmup': int(warmup),
               'params': params, 'overlays': overlays or None}
        try:
            with st.spinner('Walking through history day by day…'):
                summary = run_one(run_name.strip(), cfg, daily)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        st.success(f"Done — saved as runs/{run_name.strip()}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric('Profit', f"${summary['net_profit']:,.0f}", help=METRIC_HELP['net_profit'])
        m2.metric('Per year / robot',
                  f"{summary['ann_return_pct'] / max(cfg['gross_budget'], 1e-9):.1f}%",
                  help='Average yearly profit per robot at its backtested '
                       'size — count-independent, so runs of different team '
                       'sizes compare fairly.')
        m3.metric('Sharpe', summary['sharpe'], help=METRIC_HELP['sharpe'])
        m4.metric('Worst drop', f"{summary['max_dd_pct']:.1f}%", help=METRIC_HELP['max_dd_pct'])
        m5.metric('Churn', f"{summary['turnover_units']:.0f}", help=METRIC_HELP['turnover_units'])

        eq = pd.read_csv(os.path.join(RUNS_DIR, run_name.strip(), 'equity.csv'),
                         index_col=0, parse_dates=True)
        st.plotly_chart(equity_chart({run_name.strip(): eq}), use_container_width=True)
        st.caption('Head to **Results & Compare** to see this against the benchmarks.')


# ══════════════════════════════════════════════════════════════════════════════
# ▶ INTERACTIVE REPLAY
# ══════════════════════════════════════════════════════════════════════════════

elif page == '▶ Interactive Replay':
    st.title('▶ Interactive replay — you are the manager')
    if not timeline_name:
        st.stop()
    daily, meta = cached_timeline(timeline_name)

    st.caption('The simulation plays forward through history and **pauses every '
               'time the rules fire**. You see the evidence and make the call: '
               'swap, bench without a replacement, or overrule and keep. '
               'Your decisions steer the rest of the run.')

    if 'ir_session' not in st.session_state:
        st.session_state.ir_session = None
        st.session_state.ir_status  = None

    ses = st.session_state.ir_session

    # ── Setup form ────────────────────────────────────────────────────────
    if ses is None:
        with st.form('ir_setup'):
            st.subheader('Session setup')
            streak_mode = st.radio(
                'Count losing streaks in…', ['days', 'trades'], horizontal=True,
                help='Days = consecutive losing trading days. Trades = consecutive '
                     'losing trades from the trade history — fairer across fast and '
                     'slow robots.')
            c1, c2, c3 = st.columns(3)
            n_slots  = c1.number_input('Team size', 2, 30, 10)
            streak   = c2.slider('Bench after N losing streak', 2, 15, 5)
            dd_lim   = c3.slider('Bench after robot DD % of account', 0.5, 6.0, 2.5, 0.5)
            c4, c5, c6 = st.columns(3)
            review   = c4.slider('Review every N days', 1, 21, 5)
            use_corr = c5.checkbox('Correlation cap 0.7 on promotions', True)
            cooldown = c6.slider('Cooldown (days)', 0, 63, 21)
            c7, c8 = st.columns(3)[:2]
            use_symcap = c7.checkbox(
                'Limit robots per market', True,
                help='Diversification rule: caps how many team slots one market '
                     'can occupy, for the starting team and every promotion.')
            sym_cap = c8.number_input('Max per market', 1, 10, 3,
                                      disabled=not use_symcap)
            with st.expander('More benching rules (optional)'):
                d1, d2, d3 = st.columns(3)
                use_dollar = d1.checkbox('Bench on streak cost ($)', False,
                                         help='Fires when the current losing streak has '
                                              'cost more than the threshold in total.')
                dollar_lim = d1.number_input('Streak cost threshold ($)', 250, 20000,
                                             3000, step=250)
                use_freq = d2.checkbox('Bench on loss frequency', False,
                                       help='Fires on N losses within a recent window, '
                                            'even non-consecutive — the slow bleeder rule.')
                freq_n = d2.number_input('…this many losses', 3, 30, 8)
                freq_m = d3.number_input('…within trading days', 5, 63, 21)
            if st.form_submit_button('▶ Start session', type='primary'):
                stats = ea_stats(daily.iloc[:63])
                cap = int(sym_cap) if use_symcap else None
                portfolio = pick_top(stats, 'sharpe', int(n_slots), cap)
                tb = cached_tradebook(timeline_name) if streak_mode == 'trades' else None
                st.session_state.ir_session = InteractiveSession(
                    daily, portfolio, list(daily.columns),
                    gross=float(n_slots), review_every=int(review), warmup=63,
                    loss_streak_limit=int(streak), ea_dd_limit_pct=float(dd_lim),
                    corr_cap=0.7 if use_corr else None, cooldown_days=int(cooldown),
                    max_per_symbol=cap, streak_mode=streak_mode,
                    streak_dollar_limit=int(dollar_lim) if use_dollar else None,
                    loss_count_limit=int(freq_n) if use_freq else None,
                    loss_count_window=int(freq_m), tradebook=tb)
                st.session_state.ir_status = None
                st.rerun()
        st.stop()

    # ── Controls ──────────────────────────────────────────────────────────
    cA, cB, cC = st.columns([2, 2, 1])
    with cA:
        if ses.pending is None and st.session_state.ir_status != 'done':
            if st.button('⏵ Play to next decision', type='primary'):
                with st.spinner('Trading forward…'):
                    st.session_state.ir_status = ses.advance()
                st.rerun()
    with cB:
        if ses.pending is None and st.session_state.ir_status != 'done':
            if st.button('⏭ Auto-pilot to the end (approve everything)'):
                with st.spinner('Auto-piloting…'):
                    while ses.advance() == 'paused':
                        ses.apply_decisions(['swap'] * len(ses.pending))
                    st.session_state.ir_status = 'done'
                st.rerun()
    with cC:
        if st.button('🔄 Reset'):
            st.session_state.ir_session = None
            st.session_state.ir_status  = None
            st.rerun()

    # ── Progress + equity so far ──────────────────────────────────────────
    eq = ses.equity_frame()
    prog = ses.t / len(daily)
    when = daily.index[min(ses.t, len(daily) - 1)]
    st.progress(prog, text=f"{when:%d %b %Y} — day {ses.t:,} of {len(daily):,}")

    if not eq.empty:
        k1, k2, k3 = st.columns(3)
        k1.metric('Account', f"${ses.equity:,.0f}")
        s = ses.summary()
        k2.metric('Worst drop so far', f"{s.get('max_dd_pct', 0):.1f}%" if s else '—')
        k3.metric('Decisions made', len(ses.journal))
        st.plotly_chart(equity_chart({'your run': eq}), use_container_width=True)

    # ── Decision point ────────────────────────────────────────────────────
    if ses.pending:
        st.subheader(f"🛑 Decision point — {ses.pending_date:%d %b %Y}")
        st.caption('The rules fired. For each proposal, make your call. '
                   '"Swap" follows the rules; "Keep" overrules them.')
        decisions = []
        for i, prop in enumerate(ses.pending):
            with st.container(border=True):
                esc = lambda s: s.replace('$', r'\$')
                if prop['drop']:
                    st.markdown(f"**Bench:** `{friendly_name(prop['drop'], meta)}` — "
                                f"{esc(prop['drop_reason'])}")
                    ev = prop['evidence'].get('drop', {})
                    if ev:
                        st.caption(f"Recent window: P&L \\${ev['window_pnl']:,.0f} · "
                                   f"worst drop \\${ev['window_dd']:,.0f} · "
                                   f"losing streak {ev['loss_streak']} days")
                if prop['add']:
                    st.markdown(f"**Promote:** `{friendly_name(prop['add'], meta)}` — "
                                f"{esc(prop['add_reason'])}")
                    ev = prop['evidence'].get('add', {})
                    if ev:
                        st.caption(f"Recent window: P&L \\${ev['window_pnl']:,.0f} · "
                                   f"Sharpe {ev['sharpe']} · "
                                   f"worst drop \\${ev['window_dd']:,.0f}")
                options = ['Swap (follow the rules)']
                if prop['drop']:
                    options += ['Bench only (no replacement)', 'Keep (overrule)']
                choice = st.radio('Your call', options, key=f'ir_dec_{ses.t}_{i}',
                                  horizontal=True, label_visibility='collapsed')
                decisions.append({'Swap (follow the rules)': 'swap',
                                  'Bench only (no replacement)': 'drop_only',
                                  'Keep (overrule)': 'keep'}[choice])
        if st.button('✅ Apply my decisions & continue', type='primary'):
            ses.apply_decisions(decisions)
            with st.spinner('Trading forward…'):
                st.session_state.ir_status = ses.advance()
            st.rerun()

    # ── Finished ──────────────────────────────────────────────────────────
    if st.session_state.ir_status == 'done':
        st.success('End of history reached.')
        s = ses.summary()
        if s:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Final profit', f"${s['net_profit']:,.0f}")
            m2.metric('Per year', f"{s['ann_return_pct']:.0f}%")
            m3.metric('Sharpe', s['sharpe'])
            m4.metric('Worst drop', f"{s['max_dd_pct']:.1f}%")

    # ── Mid-session rule editor ───────────────────────────────────────────
    with st.expander('⚙️ Adjust rules mid-session'):
        st.caption('Change the team policy while the run is live — like a manager '
                   'updating the rulebook. Changes take effect from the **next '
                   'review** (a pending decision still reflects the old rules) '
                   'and every change is logged to the journal. Set a limit to 0 '
                   'to turn that rule off.')
        m1, m2, m3 = st.columns(3)
        new_mode   = m1.radio('Streak counted in', ['days', 'trades'],
                              index=0 if ses.streak_mode == 'days' else 1,
                              horizontal=True, key='ir_rule_mode')
        new_streak = m2.number_input('Losing streak limit', 0, 30,
                                     int(ses.streak_lim or 0), key='ir_rule_streak')
        new_dd     = m3.number_input('Robot DD limit (% of account)', 0.0, 10.0,
                                     float(ses.dd_lim / ses.basis * 100) if ses.dd_lim else 0.0,
                                     step=0.5, key='ir_rule_dd')
        m4, m5, m6 = st.columns(3)
        new_dollar = m4.number_input('Streak cost limit ($)', 0, 30000,
                                     int(ses.dollar_lim or 0), step=250, key='ir_rule_dollar')
        new_freqn  = m5.number_input('Loss frequency limit', 0, 40,
                                     int(ses.count_lim or 0), key='ir_rule_freqn')
        new_freqm  = m6.number_input('…within trading days', 5, 63,
                                     int(ses.count_win), key='ir_rule_freqm')
        m7, m8, m9 = st.columns(3)
        new_corr   = m7.number_input('Correlation cap', 0.0, 0.95,
                                     float(ses.corr_cap or 0.0), step=0.05, key='ir_rule_corr')
        new_cool   = m8.number_input('Cooldown (days)', 0, 63,
                                     int(ses.cooldown), key='ir_rule_cool')
        new_symcap = m9.number_input('Max robots per market', 0, 10,
                                     int(ses.max_sym or 0), key='ir_rule_symcap')
        if st.button('Apply rule changes', key='ir_rule_apply'):
            if new_mode == 'trades' and ses.tradebook is None:
                ses.tradebook = cached_tradebook(timeline_name)
            changed = ses.update_rules(
                streak_mode=new_mode,
                loss_streak_limit=int(new_streak) or None,
                ea_dd_limit_pct=float(new_dd) or None,
                streak_dollar_limit=int(new_dollar) or None,
                loss_count_limit=int(new_freqn) or None,
                loss_count_window=int(new_freqm),
                corr_cap=float(new_corr) or None,
                cooldown_days=int(new_cool),
                max_per_symbol=int(new_symcap) or None)
            if changed:
                st.success('Updated: ' + ', '.join(changed))
            else:
                st.info('No changes made.')

    # ── Current team + journal ────────────────────────────────────────────
    with st.expander('Current team', expanded=False):
        for ea in ses.active:
            st.markdown(f"- {friendly_name(ea, meta)}")
    if ses.journal:
        with st.expander(f'Decision journal ({len(ses.journal)})'):
            st.dataframe(pd.DataFrame(ses.journal), use_container_width=True,
                         hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🏁 RESULTS & COMPARE
# ══════════════════════════════════════════════════════════════════════════════

elif page == '🏁 Results & Compare':
    st.title('🏁 Results & compare')

    @st.cache_data(show_spinner=False)
    def _underwater_pct(run, mtime):
        eq = pd.read_csv(os.path.join(RUNS_DIR, run, 'equity.csv'),
                         index_col=0, parse_dates=True)['equity']
        return float((eq < eq.cummax()).mean() * 100)

    rows = {}
    for d in sorted(os.listdir(RUNS_DIR)) if os.path.isdir(RUNS_DIR) else []:
        p = os.path.join(RUNS_DIR, d, 'summary.json')
        if d.startswith('_') or not os.path.isfile(p):
            continue
        with open(p) as f:
            rows[d] = json.load(f)['summary']
        cp = os.path.join(RUNS_DIR, d, 'config.json')
        if os.path.isfile(cp):
            with open(cp) as f:
                rows[d]['risk_units'] = json.load(f).get('gross_budget')
        ep = os.path.join(RUNS_DIR, d, 'equity.csv')
        if os.path.isfile(ep):
            rows[d]['underwater_pct'] = _underwater_pct(d, os.path.getmtime(ep))
    if not rows:
        st.info('No completed runs yet — build one on the **Build a Run** page.')
        st.stop()

    df = pd.DataFrame(rows).T[['risk_units', 'net_profit', 'ann_return_pct',
                               'sharpe', 'max_dd_pct', 'underwater_pct',
                               'turnover_units', 'events']]
    df['ann_per_unit'] = np.where(df['risk_units'].astype(float) > 0,
                                  df['ann_return_pct'].astype(float)
                                  / df['risk_units'].astype(float), np.nan)
    df = df[['risk_units', 'net_profit', 'ann_per_unit', 'sharpe',
             'max_dd_pct', 'underwater_pct', 'turnover_units', 'events']]
    df = df.sort_values('sharpe', ascending=False).rename(columns={
        'risk_units': 'Risk units',
        'net_profit': 'Profit ($)',
        'ann_per_unit': 'Per year per robot (%)',
        'sharpe': 'Sharpe', 'max_dd_pct': 'Worst drop (%)',
        'underwater_pct': 'Under water (%)',
        'turnover_units': 'Churn', 'events': 'Decisions'})

    # ── Baseline ──────────────────────────────────────────────────────────
    base_default = next((n for n in ('wim_hold_37', 'bench_equal_weight')
                         if n in df.index), df.index[0])
    baseline = st.selectbox(
        'Baseline run — everything below is measured against it',
        list(df.index), index=list(df.index).index(base_default),
        help='Pick your do-nothing (or reference) portfolio: e.g. the '
             'passive hold of the same book, or the equal-weight bench. '
             'The Δ columns show what each management style adds or costs '
             'versus this baseline, and it is always drawn in the charts '
             'below.')
    df.insert(4, 'Δ Sharpe vs base',
              (df['Sharpe'] - df.loc[baseline, 'Sharpe']).round(2))
    df.insert(4, 'Δ %/robot vs base',
              (df['Per year per robot (%)']
               - df.loc[baseline, 'Per year per robot (%)']).round(2))
    st.dataframe(df, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(help=h, format='%.2f') for c, h in {
                     'Risk units'    : ('How many robots-at-full-backtested-size '
                                        'this run sums. The $100k / 5%-DD '
                                        'normalisation makes backtest figures '
                                        'directly summable, and a run\'s dollars '
                                        'are exactly that sum — so a 37-robot run '
                                        'shows ~3.7× the dollars (and dollar DD) '
                                        'of a 10-robot bench purely from summing '
                                        'more robots. Compare dollars only at the '
                                        'same count; Sharpe and per-robot columns '
                                        'are fair across counts.'),
                     'Profit ($)'    : METRIC_HELP['net_profit'],
                     'Per year per robot (%)': (
                         'Average yearly profit per robot-at-backtested-size, '
                         'on its own $100k / 5%-DD calibration — the '
                         'count-independent return figure. (Raw "% of $100k" '
                         'is misleading for multi-robot books: a 37-robot run '
                         'shows 37× this number against the same fixed base.)'),
                     'Sharpe'        : METRIC_HELP['sharpe'],
                     'Δ %/robot vs base': ('Per-robot yearly return minus the '
                                           'baseline\'s — what the management '
                                           'style adds (or costs) per robot '
                                           'against your chosen do-nothing '
                                           'reference.'),
                     'Δ Sharpe vs base': ('Sharpe minus the baseline\'s — '
                                          'positive = smoother profit per '
                                          'unit of wobble than the '
                                          'baseline.'),
                     'Worst drop (%)': METRIC_HELP['max_dd_pct'],
                     'Under water (%)': ('Share of trading days spent below '
                                         'the running equity peak. Worst drop '
                                         'is the pain; this is how LONG you '
                                         'sit in it — the endurance stat. '
                                         'Rules-managed books typically cut '
                                         'it to about half of a passive '
                                         'hold\'s.'),
                     'Churn'         : METRIC_HELP['turnover_units'],
                 }.items()})

    with st.expander('How to read this table (start here!)'):
        st.markdown("""
- **Check Risk units before comparing dollars.** The normalisation makes every
  robot's backtest figures directly summable — a run's dollars are exactly the
  sum of its robots. So a 37-robot book shows ~3.7× the profit *and* drawdown
  dollars of a 10-robot bench just from summing more robots — that is count,
  not skill. Sharpe is the fair comparison across different counts.
- **Ignore the absolute profit numbers** — this pool only contains strategies that
  already looked good on history, which flatters everything. The *comparison
  between rows* is what's meaningful.
- **Sharpe** is the fairest single column: profit per unit of day-to-day wobble.
- A style is only interesting if it beats **bench_equal_weight** (do nothing)
  convincingly — and it must crush **bench_random** (dice rolls), or its
  decisions add nothing.
- **Churn** is the hidden cost column: two styles with equal Sharpe are not equal
  if one needed 20× the swaps.
""")

    with st.expander('🔬 The deeper tests behind these results — what they are and why they matter'):
        st.markdown("""
A single backtest can look great by luck. Four extra tests guard against
fooling ourselves — all run from the engine's command line (see README):

**1. Control tests (random + do-nothing).** Every regime is compared against
*random swapping* and *equal-weight-hold-everything*. If a clever style can't
crush dice rolls and comfortably beat doing nothing, its decisions add nothing.

**2. Parameter sweeps** (`sweep_analysis.py`). We re-run a style across a whole
grid of settings (bench after 3, 4, 5… losing days × several drawdown limits).
A trustworthy style scores well across a broad *region* of settings. If only
one magic combination works, that's **curve fitting** — the settings were
fitted to history, not to anything real. Ours came out as a broad plateau.

**3. Walk-forward** (`walk_forward.py`). Choose the team and tune every setting
using ONLY older data (2018–22), then run frozen on newer data (2023–26) the
tuning never saw. This catches styles that only work in hindsight — on the
extended window it demoted every managed style below plain equal weight, the
single most honest result in this project. Bonus lesson: picking the "10 best
performers" of the calibration years builds a team of 9 Bitcoin robots — the
diversification cap prevents exactly that.

**4. Monte Carlo** (`monte_carlo.py`). History happened in one particular order;
Monte Carlo reshuffles it in blocks (weeks / months / quarters) into hundreds of
alternate histories and re-runs the style through each. The result is a *range*
of outcomes instead of one number — e.g. a ~5% team drawdown is *normal* and
~8% is unlucky-but-ordinary. Knowing that range **before** a drawdown happens
is what stops the panic-and-shelve cycle.
""")
        wf_path = os.path.join(RUNS_DIR, '_walk_forward', 'test_window_comparison.csv')
        if os.path.isfile(wf_path):
            st.markdown('**Walk-forward — test window 2023-26 (all settings frozen '
                        'using only 2018-22 data):**')
            st.dataframe(friendly_wf_table(pd.read_csv(wf_path)),
                         use_container_width=True, hide_index=True)
        mc_path = os.path.join(RUNS_DIR, '_monte_carlo', 'summary.csv')
        if os.path.isfile(mc_path):
            st.markdown('**Monte Carlo — range of outcomes across 100 reshuffled '
                        'histories per block size:**')
            st.dataframe(friendly_mc_table(pd.read_csv(mc_path)),
                         use_container_width=True, hide_index=True)
        mc2_path = os.path.join(RUNS_DIR, '_monte_carlo_streak_cost', 'summary.csv')
        if os.path.isfile(mc2_path):
            st.markdown('**Monte Carlo — streak-cost-only rule (bench when the current '
                        'losing streak has cost 1,000 dollars), same reshuffled histories:**')
            st.dataframe(friendly_mc_table(pd.read_csv(mc2_path)),
                         use_container_width=True, hide_index=True)

    with st.expander('🗂 Manage runs (delete old experiments)'):
        del_sel = st.multiselect('Runs to delete', list(df.index),
                                 help='Removes the saved run folder '
                                      '(runs/<name>: config, equity, events, '
                                      'summary). The compiled datasets are '
                                      'not touched — you can always re-run '
                                      'the same configuration.')
        sure_runs = st.checkbox('Yes, delete the selected run(s) permanently',
                                key='del_runs_confirm')
        if st.button('🗑 Delete selected runs', type='primary',
                     disabled=not (del_sel and sure_runs)):
            for name in del_sel:
                shutil.rmtree(os.path.join(RUNS_DIR, name), ignore_errors=True)
            st.success(f'{len(del_sel)} run(s) deleted.')
            st.rerun()

    comp_view = st.toggle(
        '💹 Compounding view — lot size tracks the balance', value=False,
        help='The simulations run on a FIXED balance (linear, fair comparisons, '
             'transferable thresholds). This view re-renders the same daily '
             'returns as if lots scaled with the balance — the live-compounding '
             'upper bound. Two honest notes: percentage drawdowns stay the same '
             'but their DOLLAR size grows with the balance (a normal 5% dip on '
             'a compounded account can dwarf your original stake — decide in '
             'advance that it is normal); and the curve assumes fills stay '
             'perfect as lots grow, which stops being true at scale. A stepped '
             'middle path — raising the fixed balance in deliberate jumps when '
             'you are comfortable — lands between the two lines and is the '
             'most psychologically sustainable version.')

    def apply_view(eq_df, basis=100_000):
        if not comp_view:
            return eq_df
        out = eq_df.copy()
        out['equity'] = basis * (1 + out['pnl'] / basis).cumprod()
        return out

    pick_default = [baseline] + [n for n in df.index if n != baseline][:2]
    picks = st.multiselect('Overlay equity curves',
                           list(df.index), default=pick_default,
                           help='The baseline is included by default so every '
                                'comparison is against it.')
    frames = {}
    for name in picks:
        p = os.path.join(RUNS_DIR, name, 'equity.csv')
        if os.path.isfile(p):
            frames[name] = apply_view(pd.read_csv(p, index_col=0, parse_dates=True))
    if frames:
        eqfig = equity_chart(frames)
        # Decision markers — when the management style actually acted
        any_ev = False
        for label, f in frames.items():
            evp = os.path.join(RUNS_DIR, label, 'events.csv')
            if not (os.path.isfile(evp) and os.path.getsize(evp) > 2):
                continue
            try:
                ev = pd.read_csv(evp, parse_dates=['date'])
            except (ValueError, pd.errors.ParserError):
                continue
            if ev.empty:
                continue
            per_day = ev.groupby(ev['date'].dt.normalize()).size()
            days = per_day.index.intersection(f.index)
            if not len(days):
                continue
            any_ev = True
            eqfig.add_trace(go.Scatter(
                x=days, y=f.loc[days, 'equity'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=6, opacity=0.55),
                name=f'{label} — decisions',
                customdata=per_day.loc[days],
                hovertemplate='%{x|%d %b %Y}: %{customdata} decision(s)'
                              '<extra>' + label + '</extra>'))
        st.plotly_chart(eqfig, use_container_width=True)
        if any_ev:
            st.caption('▾ markers = review days where the management style '
                       'actually acted (benched or promoted robots) — hover '
                       'for how many decisions; the Decision journal in the '
                       'drill-down lists each one. Toggle markers off via '
                       'the legend.')
        if comp_view:
            st.caption('Same strategies, same days, same percentage moves — '
                       'only the sizing rule changed. On a log axis this would '
                       'be a straight line; in dollars it is a hockey stick, '
                       'in both directions.')

        # ── Drawdown overlay — the other half of the story ────────────────
        st.subheader('Drawdown — the other half of the story')
        ddfig = go.Figure()
        for label, f in frames.items():
            peak = f['equity'].cummax()
            ddp = (peak - f['equity']) / peak * 100
            ddfig.add_trace(go.Scatter(x=f.index, y=-ddp, mode='lines',
                                       name=label,
                                       hovertemplate='%{x|%d %b %Y}<br>'
                                                     '%{y:.2f}% below peak'
                                                     '<extra>' + label +
                                                     '</extra>'))
        ddfig.add_hline(y=0, line_color='gray', line_width=1)
        ddfig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                            legend=dict(orientation='h', y=-0.2),
                            yaxis_title='% below peak')
        st.plotly_chart(ddfig, use_container_width=True)
        st.caption('Every dip is time spent under water — same runs, same '
                   'days as the curve above. Two equal profit curves are '
                   'NOT equal if one spent months down here: depth is the '
                   'pain, width is how long you had to sit through it, and '
                   'dips that line up across runs mean the styles share the '
                   'same bad weeks (no diversification between them). '
                   'Shown as % of the account\'s running peak, so the '
                   'linear and compounding views compare fairly.')

    st.subheader('Drill into one run')
    sel = st.selectbox('Run', list(df.index))
    seldir = os.path.join(RUNS_DIR, sel)
    eq = apply_view(pd.read_csv(os.path.join(seldir, 'equity.csv'),
                                index_col=0, parse_dates=True))

    dd = eq['equity'].cummax() - eq['equity']
    if comp_view:
        peak = eq['equity'].cummax()
        dd_pct = (dd / peak * 100).max()
        m1, m2, m3 = st.columns(3)
        m1.metric('Final balance (compounded)', f"${eq['equity'].iloc[-1]:,.0f}")
        m2.metric('Worst drop ($, compounded)', f"${dd.max():,.0f}",
                  help='Same percentage event as the linear view — but felt '
                       'in the balance of the day it happens.')
        m3.metric('Worst drop (%)', f"{dd_pct:.1f}%")
        if eq['equity'].iloc[-1] > 100_000 * 1000:
            st.warning('That final balance is mathematically faithful and '
                       'practically absurd — which IS the lesson: nobody '
                       'compounds an edge untouched for years. Lot sizes hit '
                       'liquidity and broker limits, fills degrade, and sane '
                       'people withdraw. Read the first year or two of this '
                       'curve as the realistic part, and the rest as a '
                       'demonstration of why capacity — not maths — is the '
                       'binding constraint.')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=-dd, fill='tozeroy', name='drawdown',
                             line=dict(color='#e05555')))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=25, b=10),
                      yaxis_title='$ below peak',
                      title='Drawdown — how far below its best the account was')
    st.plotly_chart(fig, use_container_width=True)

    ev_path = os.path.join(seldir, 'events.csv')
    if os.path.isfile(ev_path) and os.path.getsize(ev_path) > 2:
        try:
            ev = pd.read_csv(ev_path)
            if not ev.empty:
                st.subheader('Decision journal')
                st.caption('Every action this style took, dated and with its reason — '
                           'the paper trail a human manager never writes down.')
                st.dataframe(ev, use_container_width=True, hide_index=True)
        except pd.errors.EmptyDataError:
            pass
