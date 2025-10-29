# plot_variables_v3.py — single aggregated CSV, no AWS profile needed
import os
import csv
import boto3
import numpy as np
import matplotlib
# Headless by default; set SHOW=1 to display window
if os.getenv("SHOW", "0") != "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from botocore.config import Config
from botocore.exceptions import ClientError

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

# Optional: fixed resample seconds. If unset, we auto-detect min step.
RESAMPLE_SEC_ENV = os.getenv("RESAMPLE_SEC")  # e.g. "5"
# =================================

TS_COL = "ts"
V1_COL, I1_COL = "L1-N_Voltage", "L1_Current"
V2_COL, I2_COL = "L2-N_Voltage", "L2_Current"
V3_COL, I3_COL = "L3-N_Voltage", "L3_Current"

# -------- S3 client (no profile) --------
def make_s3(bucket_name: str):
    """
    Uses standard AWS credential chain:
      - Env vars (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN)
      - Shared config/credentials (ignored if you don't have them)
      - EC2/ECS/Lambda role, etc.
    """
    # Prefer AWS_REGION env; otherwise try bucket location; else default to us-east-1.
    region = os.getenv("AWS_REGION")
    session = boto3.Session(region_name=region) if region else boto3.Session()

    if not region:
        try:
            probe = session.client("s3")
            loc = probe.get_bucket_location(Bucket=bucket_name)["LocationConstraint"]
            region = loc or "us-east-1"
        except ClientError:
            region = "us-east-1"

    cfg = Config(
        region_name=region,
        retries={"max_attempts": 3, "mode": "standard"},
        connect_timeout=5,
        read_timeout=30,
        max_pool_connections=64,
    )
    print(f"[boto3] Using region={region!r} (no profile)")
    return session.client("s3", config=cfg)

s3 = make_s3(BUCKET)

# -------- Helpers --------
def today_local_str():
    tz = get_tz(LOCAL_TZ_NAME)
    return datetime.now(tz).strftime("%Y-%m-%d")

def key_for_local_date_csv(date_str_local: str):
    """Build s3 key like Carmelo/YYYY/MM/DD.csv using the LOCAL (Atlantic/Canary) date."""
    dt = datetime.fromisoformat(date_str_local)
    y = dt.strftime("%Y")
    m = dt.strftime("%m")
    d = dt.strftime("%d")
    return f"{PREFIX}{y}/{m}/{d}.csv"

def to_ms_from_num(x) -> int:
    """Accept seconds or ms; normalize to ms."""
    t = int(float(x))
    return t if t >= 10**12 else t * 1000

def _idx(header, name):
    try:
        return header.index(name)
    except ValueError:
        return None

# -------- CSV reader (single aggregated file) --------
def fetch_all_from_csv_key(key: str):
    """Read the whole CSV and return arrays for ts, V/I per line."""
    print(f"[S3] Downloading: s3://{BUCKET}/{key}")
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    text = obj["Body"].read().decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("CSV empty")

    reader = csv.reader(lines)
    header = next(reader, None)
    if header is None:
        raise RuntimeError("CSV missing header")

    ts_i = _idx(header, TS_COL)
    v1_i, i1_i = _idx(header, V1_COL), _idx(header, I1_COL)
    v2_i, i2_i = _idx(header, V2_COL), _idx(header, I2_COL)
    v3_i, i3_i = _idx(header, V3_COL), _idx(header, I3_COL)
    needed = [ts_i, v1_i, i1_i, v2_i, i2_i, v3_i, i3_i]
    if any(i is None for i in needed):
        raise RuntimeError(
            f"CSV missing required columns: {TS_COL}, "
            f"{V1_COL}/{I1_COL}, {V2_COL}/{I2_COL}, {V3_COL}/{I3_COL}"
        )

    ts_ms, V1, I1, V2, I2, V3, I3 = [], [], [], [], [], [], []
    bad = 0
    for row in reader:
        try:
            t = to_ms_from_num(row[ts_i])
            v1, i1 = float(row[v1_i]), float(row[i1_i])
            v2, i2 = float(row[v2_i]), float(row[i2_i])
            v3, i3 = float(row[v3_i]), float(row[i3_i])
        except Exception:
            bad += 1
            continue
        ts_ms.append(t); V1.append(v1); I1.append(i1); V2.append(v2); I2.append(i2); V3.append(v3); I3.append(i3)

    if not ts_ms:
        raise RuntimeError("No valid data rows parsed from CSV.")
    if bad:
        print(f"[CSV] Skipped {bad} malformed rows")

    order = np.argsort(np.array(ts_ms, dtype=np.int64))
    ts_ms = np.array(ts_ms, dtype=np.int64)[order]
    V1 = np.array(V1, dtype=float)[order]
    I1 = np.array(I1, dtype=float)[order]
    V2 = np.array(V2, dtype=float)[order]
    I2 = np.array(I2, dtype=float)[order]
    V3 = np.array(V3, dtype=float)[order]
    I3 = np.array(I3, dtype=float)[order]
    return ts_ms, V1, I1, V2, I2, V3, I3

# -------- Resampling --------
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
    print(f"Bucket={BUCKET} Prefix={PREFIX} Date(local)={date_local} TZ={LOCAL_TZ_NAME}")

    # Aggregated CSV key for the LOCAL date (e.g., Carmelo/2025/10/28.csv)
    key = key_for_local_date_csv(date_local)
    print(f"[Key] Using aggregated file: s3://{BUCKET}/{key}")

    # Read all rows from the single CSV
    ts_ms, V1, I1, V2, I2, V3, I3 = fetch_all_from_csv_key(key)

    # Keep only samples within the local day window (defensive)
    tz_local = get_tz(LOCAL_TZ_NAME)
    start_local = datetime.fromisoformat(date_local).replace(tzinfo=tz_local)
    end_local = start_local + timedelta(days=1)
    start_ms = int(start_local.astimezone(timezone.utc).timestamp() * 1000)
    end_ms = int(end_local.astimezone(timezone.utc).timestamp() * 1000)

    mask = (ts_ms >= start_ms) & (ts_ms < end_ms)
    ts_ms = ts_ms[mask]
    V1, I1 = V1[mask], I1[mask]
    V2, I2 = V2[mask], I2[mask]
    V3, I3 = V3[mask], I3[mask]
    print(f"[Filter] Rows in local-day window: {ts_ms.size}")

    if ts_ms.size == 0:
        print("No data for the requested day.")
        return

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

    # Convert to local time & plot
    dts_local = to_local_datetimes(ts_ms_res)
    plot_power(dts_local, P1r, P2r, P3r, Tr, date_local, step_sec)
    plot_voltages(dts_local, V1r, V2r, V3r, date_local, step_sec)
    plot_currents(dts_local, I1r, I2r, I3r, date_local, step_sec)

    # Energy from total
    kwh = daily_energy_kwh(dts_local, Tr)
    print(f"[Energy] Estimated daily apparent energy (V×I sum) for {date_local}: {kwh:.2f} kWh")

if __name__ == "__main__":
    main()
