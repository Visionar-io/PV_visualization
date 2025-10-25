# plot_daily_kw_numpy_3phase_lines_auto_vi.py
import os
import re
import csv
import math
import boto3
import numpy as np
import matplotlib
# Headless by default; set SHOW=1 to display window
if os.getenv("SHOW", "0") != "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.config import Config

# -------- Timezone helpers --------
try:
    from zoneinfo import ZoneInfo
    def get_tz(name): return ZoneInfo(name)
except Exception:
    from dateutil import tz
    def get_tz(name): return tz.gettz(name)

# ============ CONFIG ============
BUCKET = os.getenv("BUCKET", "pm-metering")
PREFIX = os.getenv("PREFIX", "Carmelo/").rstrip("/") + "/"
DATE_STR = os.getenv("DATE")  # YYYY-MM-DD
LOCAL_TZ_NAME = os.getenv("TZ", "Atlantic/Canary")
PROFILE = os.getenv("AWS_PROFILE", "pm")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "32"))
MAX_KEYS = int(os.getenv("MAX_KEYS", "0"))

# Resampling: if RESAMPLE_SEC>0 use that; otherwise auto-detect min Δt
RESAMPLE_SEC_ENV = os.getenv("RESAMPLE_SEC")  # e.g. "5"; unset/"" => auto
# ================================

TS_COL = "ts"
V1_COL, I1_COL = "L1-N_Voltage", "L1_Current"
V2_COL, I2_COL = "L2-N_Voltage", "L2_Current"
V3_COL, I3_COL = "L3-N_Voltage", "L3_Current"

TS_RE = re.compile(r"/(\d{10,13})\.txt$")

# -------- S3 client --------
def make_s3(bucket_name: str):
    session = boto3.Session(profile_name=PROFILE)
    region = os.getenv("AWS_REGION")
    if not region:
        s3_global = session.client("s3")
        loc = s3_global.get_bucket_location(Bucket=bucket_name)["LocationConstraint"]
        region = loc or "us-east-1"
    cfg = Config(
        region_name=region,
        retries={"max_attempts": 3, "mode": "standard"},
        connect_timeout=5,
        read_timeout=15,
        max_pool_connections=max(64, MAX_WORKERS * 2),
    )
    print(f"[boto3] Using profile={PROFILE!r} region={region!r}")
    return session.client("s3", config=cfg)

s3 = make_s3(BUCKET)

# -------- Helpers --------
def today_local_str():
    tz = get_tz(LOCAL_TZ_NAME)
    return datetime.now(tz).strftime("%Y-%m-%d")

def local_day_bounds_ms(date_str_local: str):
    tz_local = get_tz(LOCAL_TZ_NAME)
    start_local = datetime.fromisoformat(date_str_local).replace(tzinfo=tz_local)
    end_local = start_local + timedelta(days=1)
    return int(start_local.timestamp() * 1000), int(end_local.timestamp() * 1000)

def list_keys_for_local_window(bucket, root_prefix, date_str_local):
    tz_local = get_tz(LOCAL_TZ_NAME)
    start_local = datetime.fromisoformat(date_str_local).replace(tzinfo=tz_local)
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)

    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    day = start_utc.date()
    while day <= end_utc.date():
        day_prefix = f"{root_prefix}{day.strftime('%Y/%m/%d/')}"
        print(f"[S3] Listing: s3://{bucket}/{day_prefix}")
        for page in paginator.paginate(Bucket=bucket, Prefix=day_prefix, PaginationConfig={"PageSize": 1000}):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".txt"):
                    keys.append(obj["Key"])
        day = (datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)).date()
    print(f"[S3] Total files discovered: {len(keys)}")
    return keys

def ts_from_key(key: str):
    m = TS_RE.search(key)
    if not m:
        return None
    t = int(m.group(1))
    return t if t >= 10**12 else t * 1000

# -------- CSV parsing: return V & I (we derive power later) --------
def _idx(header, name):
    try:
        return header.index(name)
    except ValueError:
        return None

def fetch_vi_from_key(key: str):
    """Return (ts_ms, V1, I1, V2, I2, V3, I3) or None."""
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        text = obj["Body"].read().decode("utf-8", errors="replace").strip()
        if not text:
            return None
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return None

        header = next(csv.reader([lines[0]]))
        data = next(csv.reader([lines[-1]]))

        ts_i = _idx(header, TS_COL)
        v1_i, i1_i = _idx(header, V1_COL), _idx(header, I1_COL)
        v2_i, i2_i = _idx(header, V2_COL), _idx(header, I2_COL)
        v3_i, i3_i = _idx(header, V3_COL), _idx(header, I3_COL)
        if ts_i is None or None in (v1_i, i1_i, v2_i, i2_i, v3_i, i3_i):
            return None

        ts_ms = int(float(data[ts_i]))
        V1, I1 = float(data[v1_i]), float(data[i1_i])
        V2, I2 = float(data[v2_i]), float(data[i2_i])
        V3, I3 = float(data[v3_i]), float(data[i3_i])
        return ts_ms, V1, I1, V2, I2, V3, I3
    except Exception:
        return None

# -------- Resampling at minimum Δt --------
def auto_min_step_seconds(ts_ms: np.ndarray) -> int:
    if ts_ms.size < 2:
        return 1
    d = np.diff(ts_ms.astype(np.int64))
    d = d[d > 0]
    if d.size == 0:
        return 1
    min_ms = int(np.min(d))
    return max(1, int(round(min_ms / 1000.0)))

def resample_fixed_step(ts_ms: np.ndarray, arrays: list[np.ndarray], step_sec: int):
    if ts_ms.size == 0:
        return ts_ms, arrays
    t0_ms = int(ts_ms[0])
    step_ms = step_sec * 1000
    idx = (ts_ms - t0_ms) // step_ms
    counts = {}
    sums = [dict() for _ in arrays]
    for k in idx:
        counts[k] = counts.get(k, 0) + 1
    for k, *vals in zip(idx, *arrays):
        for i, v in enumerate(vals):
            sums[i][k] = sums[i].get(k, 0.0) + float(v)
    k_sorted = sorted(counts.keys())
    out_arrays = [np.array([sums[i][k] / counts[k] for k in k_sorted], dtype=float) for i in range(len(arrays))]
    ts_out = np.array([t0_ms + k * step_ms for k in k_sorted], dtype=np.int64)
    return ts_out, out_arrays

# -------- Conversions, energy, plotting --------
def to_local_datetimes(ts_ms_arr):
    tz_local = get_tz(LOCAL_TZ_NAME)
    return np.array([datetime.fromtimestamp(int(t)/1000.0, tz=timezone.utc).astimezone(tz_local) for t in ts_ms_arr], dtype=object)

def daily_energy_kwh(dts_local, kw):
    if len(kw) < 2:
        return float("nan")
    secs = np.array([dt.timestamp() for dt in dts_local], dtype=float)
    hours = (secs - secs[0]) / 3600.0
    return float(np.trapz(kw, hours))

def plot_power(dts_local, p1, p2, p3, total, date_str_local, step_sec):
    title = f"3-Phase Power – {date_str_local} ({LOCAL_TZ_NAME}) – step={step_sec}s"
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dts_local, p1, label="L1", linewidth=1)
    ax.plot(dts_local, p2, label="L2", linewidth=1)
    ax.plot(dts_local, p3, label="L3", linewidth=1)
    ax.plot(dts_local, total, label="Total", color="black", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (kW = V×I)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    out = f"plot_{date_str_local}_lines_step{step_sec}s.png"
    plt.savefig(out, dpi=150)
    if os.getenv("SHOW", "0") == "1":
        plt.show()
    print(f"[Plot] Saved {out}")

def plot_voltages(dts_local, V1, V2, V3, date_str_local, step_sec):
    title = f"Voltages – {date_str_local} ({LOCAL_TZ_NAME}) – step={step_sec}s"
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dts_local, V1, label="L1-N Voltage", linewidth=1)
    ax.plot(dts_local, V2, label="L2-N Voltage", linewidth=1)
    ax.plot(dts_local, V3, label="L3-N Voltage", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    out = f"plot_{date_str_local}_voltages_step{step_sec}s.png"
    plt.savefig(out, dpi=150)
    print(f"[Plot] Saved {out}")

def plot_currents(dts_local, I1, I2, I3, date_str_local, step_sec):
    title = f"Currents – {date_str_local} ({LOCAL_TZ_NAME}) – step={step_sec}s"
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dts_local, I1, label="L1 Current", linewidth=1)
    ax.plot(dts_local, I2, label="L2 Current", linewidth=1)
    ax.plot(dts_local, I3, label="L3 Current", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Current (A)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    out = f"plot_{date_str_local}_currents_step{step_sec}s.png"
    plt.savefig(out, dpi=150)
    print(f"[Plot] Saved {out}")

# -------- Main --------
def main():
    date_local = DATE_STR or today_local_str()
    start_ms, end_ms = local_day_bounds_ms(date_local)
    print(f"Bucket={BUCKET} Prefix={PREFIX} Date(local)={date_local} TZ={LOCAL_TZ_NAME}")

    # List keys for the UTC folders overlapping the local day
    all_keys = list_keys_for_local_window(BUCKET, PREFIX, date_local)
    keys = [k for k in all_keys if (t := ts_from_key(k)) and start_ms <= t < end_ms]
    print(f"[Filter] Keys in local-day window: {len(keys)}")
    if MAX_KEYS > 0 and len(keys) > MAX_KEYS:
        keys = keys[:MAX_KEYS]
        print(f"[Filter] Capped to first {MAX_KEYS} keys.")

    if not keys:
        print("No files found.")
        return

    # Fetch & parse in parallel
    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_vi_from_key, k): k for k in keys}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r:
                rows.append(r)
            if i % 200 == 0 or i == len(keys):
                print(f"[Fetch] {i}/{len(keys)} done")

    if not rows:
        print("No valid data parsed.")
        return

    rows.sort(key=lambda x: x[0])
    ts_ms = np.array([r[0] for r in rows], dtype=np.int64)
    V1 = np.array([r[1] for r in rows], dtype=float)
    I1 = np.array([r[2] for r in rows], dtype=float)
    V2 = np.array([r[3] for r in rows], dtype=float)
    I2 = np.array([r[4] for r in rows], dtype=float)
    V3 = np.array([r[5] for r in rows], dtype=float)
    I3 = np.array([r[6] for r in rows], dtype=float)

    # Compute per-phase and total power (kW = V×I / 1000)
    P1 = (V1 * I1) / 1000.0
    P2 = (V2 * I2) / 1000.0
    P3 = (V3 * I3) / 1000.0
    T  = P1 + P2 + P3

    # Determine resample step
    if RESAMPLE_SEC_ENV and RESAMPLE_SEC_ENV.isdigit() and int(RESAMPLE_SEC_ENV) > 0:
        step_sec = int(RESAMPLE_SEC_ENV)
        print(f"[Resample] Using fixed step_sec from env: {step_sec}s")
    else:
        step_sec = auto_min_step_seconds(ts_ms)
        print(f"[Resample] Auto-detected min step: {step_sec}s")

    # Resample onto fixed grid
    ts_ms_res, [P1r, P2r, P3r, Tr, V1r, V2r, V3r, I1r, I2r, I3r] = resample_fixed_step(
        ts_ms, [P1, P2, P3, T, V1, V2, V3, I1, I2, I3], step_sec=step_sec
    )

    # Convert to local time
    dts_local = to_local_datetimes(ts_ms_res)

    # Plot & save three separate figures
    plot_power(dts_local, P1r, P2r, P3r, Tr, date_local, step_sec)
    plot_voltages(dts_local, V1r, V2r, V3r, date_local, step_sec)
    plot_currents(dts_local, I1r, I2r, I3r, date_local, step_sec)

    # Energy from total
    kwh = daily_energy_kwh(dts_local, Tr)
    print(f"[Energy] Estimated daily apparent energy (V×I sum) for {date_local}: {kwh:.2f} kWh")

if __name__ == "__main__":
    main()
