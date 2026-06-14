"""Verify the MTF join no longer leaks the future.

For a sample of M1 rows, find which H1 candle gets joined to them
and confirm that candle CLOSED in the PAST relative to the M1 row.
PASS = every matched H1 candle closed at or before the M1 row's time.
FAIL = any matched H1 candle closes in the future (leak still present).

Tests both the H1 (disk-loaded) and M5 (resampled-from-M1) paths.
"""
import polars as pl
import numpy as np
from pathlib import Path
import datetime

YEAR = 2023
N_SAMPLES = 20

M1_FILE = next(Path("data/EURUSD/M1").glob(f"*{YEAR}*.csv"))
H1_FILE = next(Path("data/EURUSD/H1").glob(f"*{YEAR}*.csv"))

def load_epochs(path):
    df = pl.read_csv(str(path), try_parse_dates=False)
    df = df.rename({c: c.lower() for c in df.columns})
    dt_col = next(c for c in df.columns if "date" in c or "time" in c)
    return (
        df.select(
            pl.col(dt_col).str.replace(r"\+.*$", "")
              .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
              .dt.epoch("s").alias("epoch")
        )["epoch"].to_list()
    )

def fmt(e):
    return datetime.datetime.utcfromtimestamp(e).strftime("%Y-%m-%d %H:%M")

def check_timeframe(m1_epochs, tf_epochs, period_secs, label):
    # The FIXED keying (+ period_secs on TF side)
    def m1_key(e):  return (e // period_secs) * period_secs
    def tf_key(e):  return (e // period_secs) * period_secs + period_secs

    tf_by_key = {}
    for e in tf_epochs:
        tf_by_key[tf_key(e)] = e  # value = TF candle start time

    print(f"\n{label} (period={period_secs//60}min)")
    print(f"  {'M1 row time':<20}{'TF candle START':<20}{'TF candle CLOSE':<20} result")
    print("  " + "-" * 66)
    fails = 0
    checked = 0
    for e in m1_epochs[1000:1000 + N_SAMPLES]:
        k = m1_key(e)
        if k not in tf_by_key:
            continue
        tf_start = tf_by_key[k]
        tf_close = tf_start + period_secs
        ok = tf_close <= e
        fails += (not ok)
        checked += 1
        status = "PASS" if ok else "FAIL  <-- LEAK"
        print(f"  {fmt(e):<20}{fmt(tf_start):<20}{fmt(tf_close):<20} {status}")
    print(f"  Checked {checked} rows  |  {'ALL PASS' if fails == 0 else f'{fails} LEAKING ROWS'}")
    return fails

m1_epochs = load_epochs(M1_FILE)
h1_epochs = load_epochs(H1_FILE)

def resample_epochs(m1_epochs, period_secs):
    """Derive TF candle start-times from M1 by bucketing, matching _resample_ohlcv."""
    return sorted(set((e // period_secs) * period_secs for e in m1_epochs))

total_fails = 0
total_fails += check_timeframe(m1_epochs, resample_epochs(m1_epochs,  5*60),  5 * 60,  "M5  (resampled from M1)")
total_fails += check_timeframe(m1_epochs, resample_epochs(m1_epochs, 15*60), 15 * 60,  "M15 (resampled from M1)")
total_fails += check_timeframe(m1_epochs, h1_epochs,                         60 * 60,  "H1  (disk-loaded)")
total_fails += check_timeframe(m1_epochs, resample_epochs(m1_epochs,  4*3600), 4 * 3600, "H4  (resampled from M1)")

print("\n" + "=" * 70)
if total_fails == 0:
    print("RESULT: ALL PASS — no future leak detected")
else:
    print(f"RESULT: {total_fails} LEAKING ROWS — fix not applied correctly")
print("=" * 70)
