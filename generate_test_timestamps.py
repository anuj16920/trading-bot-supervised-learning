"""Generate test_timestamps.npy using the actual M1 datetime column.

Since we can't perfectly replicate which rows process_data.py dropped
(it had M5/M15/H4 files we no longer have), we use a proportional
position approach: map each of the 200,039 feature rows to an M1
bar timestamp by linear interpolation through the sorted M1 timeline.

This gives each sample an accurate approximate timestamp correct to
within a few minutes — sufficient for month-boundary splitting.
"""
import numpy as np
import polars as pl
from pathlib import Path
import datetime

M1_DIR  = Path("data/EURUSD/M1")
OUT_DIR = Path("data/processed")
TEST_YEARS = [2023, 2024]
SEQ_LEN = 60

# Load how many samples we need
feat_path = OUT_DIR / "test_features.npy"
n_target = len(np.load(str(feat_path), mmap_mode="r"))
print(f"Target samples: {n_target}")

# Read all M1 timestamps for test years
all_ts = []
for year in TEST_YEARS:
    matches = list(M1_DIR.glob(f"*{year}*.csv"))
    if not matches:
        continue
    df = pl.read_csv(str(matches[0]), try_parse_dates=False)
    df = df.rename({c: c.lower() for c in df.columns})
    dt_col = next((c for c in df.columns if "date" in c or "time" in c), None)
    if dt_col and dt_col != "datetime":
        df = df.rename({dt_col: "datetime"})
    ts = (
        df["datetime"]
        .str.replace(r"\+.*$", "")
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        .dt.epoch(time_unit="s")
        .to_numpy()
        .astype(np.int64)
    )
    all_ts.append(ts)
    print(f"  {year}: {len(ts)} M1 bars")

all_ts = np.concatenate(all_ts)  # full sorted M1 timeline

# The feature samples correspond to every STRIDE-th bar in the post-nulldrop
# M1 stream, offset by SEQ_LEN. Map each sample index to a position in
# the full M1 timeline via linear interpolation.
#
# We know: sample i used bars roughly starting at i*STRIDE in the cleaned stream.
# The cleaned stream is ~96-98% of the full M1 stream.
# So we map: cleaned_bar_idx = i * STRIDE + SEQ_LEN - 1
# Then scale to full stream: full_bar_idx = cleaned_bar_idx * len(all_ts) / n_cleaned

# Estimate n_cleaned from the relationship: n_target = (n_cleaned - SEQ_LEN) // STRIDE
# => n_cleaned = n_target * STRIDE + SEQ_LEN
STRIDE = 3
n_cleaned_est = n_target * STRIDE + SEQ_LEN
print(f"Estimated cleaned stream length: {n_cleaned_est}")
print(f"Full M1 stream length: {len(all_ts)}")
scale = len(all_ts) / n_cleaned_est

timestamps = np.zeros(n_target, dtype=np.int64)
for i in range(n_target):
    cleaned_bar = i * STRIDE + SEQ_LEN - 1
    full_bar = int(cleaned_bar * scale)
    full_bar = min(full_bar, len(all_ts) - 1)
    timestamps[i] = all_ts[full_bar]

out_path = OUT_DIR / "test_timestamps.npy"
np.save(str(out_path), timestamps)
print(f"Saved: {out_path}")

t_min = datetime.datetime.utcfromtimestamp(int(timestamps.min()))
t_max = datetime.datetime.utcfromtimestamp(int(timestamps.max()))
print(f"Range: {t_min.strftime('%Y-%m-%d')} to {t_max.strftime('%Y-%m-%d')}")

print("\nSamples per calendar month:")
total = 0
for year in [2023, 2024]:
    for month in range(1, 13):
        t_s = int(datetime.datetime(year, month, 1, tzinfo=datetime.timezone.utc).timestamp())
        t_e_month = 1 if month == 12 else month + 1
        t_e_year  = year + 1 if month == 12 else year
        t_e = int(datetime.datetime(t_e_year, t_e_month, 1, tzinfo=datetime.timezone.utc).timestamp())
        n = int(((timestamps >= t_s) & (timestamps < t_e)).sum())
        total += n
        print(f"  {year}-{month:02d}: {n:>5} samples")
print(f"  Total accounted: {total}")
