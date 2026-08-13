"""
fetch_reference_data.py
=======================
Download daily reference-instrument history (Yahoo Finance) used to build the
regime matrix. Saves reference/reference_prices.csv (wide, close prices).

Primary source: Yahoo Finance via yfinance. If Yahoo drops a series, the
agreed fallback is Dukascopy's free historical data (dukascopy-python) —
swap the failing ticker's loader, keep the same output format.

Usage: python fetch_reference_data.py
"""

import os

import pandas as pd
import yfinance as yf

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(ENGINE_DIR, 'reference')

# label -> Yahoo ticker
TICKERS = {
    'DXY'    : 'DX-Y.NYB',   # US dollar index
    'VIX'    : '^VIX',       # equity volatility
    'SP500'  : '^GSPC',      # equities
    'GOLD'   : 'GC=F',       # gold futures
    'BTC'    : 'BTC-USD',    # bitcoin
    'US10Y'  : '^TNX',       # 10-year yield
    'OIL'    : 'CL=F',       # WTI crude
    'EURUSD' : 'EURUSD=X',
}

# EA market -> Yahoo ticker, for instrument-level history (cross-validation
# against what the robots actually traded). Dukascopy fallback: the School
# Run App's data/fetcher.py downloads bi5 directly (symbols like USA30IDXUSD,
# DEUIDXEUR) if any of these prove unreliable.
INSTRUMENT_TICKERS = {
    'XAUUSD': 'GC=F',    'XAGUSD': 'SI=F',    'XTIUSD': 'CL=F',
    'BTCUSD': 'BTC-USD', 'US30'  : '^DJI',    'US500' : '^GSPC',
    'USTEC' : '^NDX',    'DE40'  : '^GDAXI',
    'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'USDJPY=X',
    'AUDUSD': 'AUDUSD=X', 'NZDUSD': 'NZDUSD=X', 'USDCAD': 'USDCAD=X',
    'AUDJPY': 'AUDJPY=X', 'CHFJPY': 'CHFJPY=X', 'EURAUD': 'EURAUD=X',
    'EURJPY': 'EURJPY=X', 'EURNZD': 'EURNZD=X', 'GBPJPY': 'GBPJPY=X',
}

START = '2019-06-01'         # extra runway so 200-day averages exist by 2020
END   = None                 # today


def fetch_set(tickers, out_name):
    frames = {}
    for label, ticker in tickers.items():
        # period='max' then slice — Yahoo's start/end query intermittently
        # returns near-empty history for index tickers like ^TNX
        df = yf.download(ticker, period='max', progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            df = df.loc[START:]
        if df is None or df.empty:
            print(f"  WARN: no data for {label} ({ticker}) — use the "
                  f"Dukascopy fallback for this series")
            continue
        close = df['Close']
        if isinstance(close, pd.DataFrame):     # yfinance MultiIndex quirk
            close = close.iloc[:, 0]
        frames[label] = close
        print(f"  {label:7s} {ticker:10s} {len(close)} rows "
              f"{close.index[0].date()} -> {close.index[-1].date()}")
    prices = pd.DataFrame(frames).sort_index()
    prices.index.name = 'date'
    out = os.path.join(OUT_DIR, out_name)
    prices.to_csv(out)
    print(f"  Saved {prices.shape[0]} rows x {prices.shape[1]} series -> {out}\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Reference (regime) series:")
    fetch_set(TICKERS, 'reference_prices.csv')
    print("Instrument-level series (EA markets):")
    fetch_set(INSTRUMENT_TICKERS, 'instrument_prices.csv')


if __name__ == '__main__':
    main()
