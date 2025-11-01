# hourly_kwh_from_counters.py — single aggregated CSV, no AWS profile needed
import os
import csv
import boto3
import numpy as np
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
DATE_STR = os.getenv("DATE")  # YYYY-MM-DD (local)
LOCAL_TZ_NAME = os.getenv("TZ", "Atlantic/Canary")
OUT_DIR = os.getenv("OUT_DIR", ".")
# =================================

TS_COL = "ts"

# Energy counter names (present in your CSVs)
PHASE_KWH_COLS = ["L1_Total_kWh", "L2_Total_kWh", "L3_Total_kWh"]
IMPORT_KWH_COLS = ["Import_Active_Energy_resettable", "Import_Active_Energy", "Total_Import_kWh"]
EXPORT_KWH_COLS = ["Export_Active_Energy_resettable", "Export_Active_Energy", "Total_Export_kWh"]

# Optional per-phase import/export & active power cols (present in your file)
PHASE_IMPORT_COLS = ["L1_Import_kWh", "L2_Import_kWh", "L3_Import_kWh"]
PHASE_EXPORT_COLS = ["L1_Export_kWh", "L2_Export_kWh", "L3_Export_kWh"]
PHASE_ACTIVE_P_COLS = ["L1_Active_Power", "L2_Active_Power", "L3_Active_Power"]

# -------- S3 client (no profile) --------
def make_s3(bucket_name: str):
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

def _available(header, names):
    """Return [name for name in names if present in header] preserving the given order."""
    return [n for n in names if n in header]

def _hour_floor_local(ts_ms: int, tz_local) -> datetime:
    """Return local datetime floored to hour for given UTC ms epoch."""
    dt_local = datetime.fromtimestamp(ts_ms/1000.0, tz=timezone.utc).astimezone(tz_local)
    return dt_local.replace(minute=0, second=0, microsecond=0)

def _safe_sum(arr):
    if arr is None:
        return float("nan")
    s = np.nansum(arr)
    return float(s) if np.isfinite(s) else float("nan")

# -------- Hourly kWh from counters --------
def hourly_kwh_from_counters(key: str, start_ms: int, end_ms: int):
    """
    Reads the single aggregated CSV from s3://BUCKET/key and computes hourly kWh by subtracting
    cumulative counters each hour. Returns (rows, header, raw_csv_text).
    """
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
    if ts_i is None:
        raise RuntimeError(f"CSV missing required column {TS_COL!r}")

    # Which energy columns are available?
    phases = _available(header, PHASE_KWH_COLS)
    imports = _available(header, IMPORT_KWH_COLS)
    exports = _available(header, EXPORT_KWH_COLS)

    if not phases and not imports and not exports:
        raise RuntimeError("No energy counters found (expected *_Total_kWh or Import/Export counters).")

    # Build per-column hour->last_value maps (rows inside local-day window)
    tz_local = get_tz(LOCAL_TZ_NAME)
    percol_hour_last = {name: {} for name in (phases + imports + exports)}

    bad = 0
    for row in reader:
        try:
            t_ms = to_ms_from_num(row[ts_i])
        except Exception:
            bad += 1
            continue
        if not (start_ms <= t_ms < end_ms):
            continue
        hour_dt = _hour_floor_local(t_ms, tz_local)
        for name in percol_hour_last.keys():
            j = _idx(header, name)
            if j is None:
                continue
            try:
                v = float(row[j])
            except Exception:
                v = float("nan")
            if np.isfinite(v):
                percol_hour_last[name][hour_dt] = v
    if bad:
        print(f"[CSV] Skipped {bad} malformed rows")

    # Hours present in any column
    all_hours = set()
    for d in percol_hour_last.values():
        all_hours.update(d.keys())
    if not all_hours:
        return [], [], text

    hours_sorted = sorted(all_hours)

    # Sequence per column aligned to hours_sorted
    def seq_for(name):
        d = percol_hour_last.get(name, {})
        return [d.get(h, float("nan")) for h in hours_sorted]

    # Hourly deltas
    hourly_cols = {}
    for name in phases + imports + exports:
        vals = np.asarray(seq_for(name), dtype=float)
        # forward-fill
        for k in range(1, len(vals)):
            if not np.isfinite(vals[k]):
                vals[k] = vals[k-1]
        diffs = np.diff(vals, prepend=np.nan)
        diffs[0] = np.nan
        diffs = np.clip(diffs, a_min=0.0, a_max=None)
        hourly_cols[name] = diffs

    # Combine into totals
    total_from_phases = None
    if phases:
        stack = np.vstack([hourly_cols.get(n, np.full(len(hours_sorted), np.nan)) for n in phases])
        total_from_phases = np.nansum(stack, axis=0)
    imp_series = None
    exp_series = None
    if imports:
        imp_series = np.nansum(np.vstack([hourly_cols.get(n, np.full(len(hours_sorted), np.nan)) for n in imports]), axis=0)
    if exports:
        exp_series = np.nansum(np.vstack([hourly_cols.get(n, np.full(len(hours_sorted), np.nan)) for n in exports]), axis=0)
    net_series = None
    if imp_series is not None and exp_series is not None:
        net_series = np.clip(imp_series - exp_series, a_min=0.0, a_max=None)

    # Output header/rows
    out_header = ["HourStart"]
    for ph in phases:
        out_header.append(f"{ph.replace('_Total_kWh','')}_kWh" if ph.endswith("_Total_kWh") else f"{ph}_kWh")
    if phases:
        out_header.append("Total_kWh")
    if imports:
        out_header.append("Import_kWh")
    if exports:
        out_header.append("Export_kWh")
    if net_series is not None:
        out_header.append("Net_kWh")

    rows = []
    for i, h in enumerate(hours_sorted):
        row = [h.strftime("%Y-%m-%d %H:00")]
        for ph in phases:
            v = hourly_cols.get(ph, [np.nan]*len(hours_sorted))[i]
            row.append("" if not np.isfinite(v) else f"{v:.6f}")
        if phases:
            tv = total_from_phases[i] if total_from_phases is not None else np.nan
            row.append("" if not np.isfinite(tv) else f"{tv:.6f}")
        if imports:
            iv = imp_series[i] if imp_series is not None else np.nan
            row.append("" if not np.isfinite(iv) else f"{iv:.6f}")
        if exports:
            ev = exp_series[i] if exp_series is not None else np.nan
            row.append("" if not np.isfinite(ev) else f"{ev:.6f}")
        if net_series is not None:
            nv = net_series[i]
            row.append("" if not np.isfinite(nv) else f"{nv:.6f}")
        rows.append(row)

    # Summary
    print("[Summary]")
    if phases:
        print(f"  Sum of hourly Total_kWh (phases): {_safe_sum(total_from_phases):.3f} kWh")
    if imports:
        print(f"  Sum of hourly Import_kWh: {_safe_sum(imp_series):.3f} kWh")
    if exports:
        print(f"  Sum of hourly Export_kWh: {_safe_sum(exp_series):.3f} kWh")
    if net_series is not None:
        print(f"  Sum of hourly Net_kWh: {_safe_sum(net_series):.3f} kWh")

    return rows, out_header, text  # raw CSV text for diagnosis

# -------- Diagnose which phase caused Export (pure-Python integration, no NumPy arrays) --------
def diagnose_export_sources(csv_text: str, start_ms: int, end_ms: int):
    """
    Stream through CSV rows and compute, per-phase:
      - % of negative active-power samples
      - Import/Export energy from power trace (kWh) via trapezoid
      - Import/Export energy from per-phase meters (kWh) via end-start
    Uses only Python floats/lists; avoids NumPy array construction.
    """
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    reader = csv.reader(lines)
    header = next(reader, None)
    if header is None:
        print("[Diagnose] Missing header.")
        return

    name_to_idx = {n: i for i, n in enumerate(header)}
    ts_i = name_to_idx.get(TS_COL)
    if ts_i is None:
        print("[Diagnose] ts column missing; skip.")
        return

    phases = ["L1", "L2", "L3"]
    ap_idx  = {ph: name_to_idx.get(f"{ph}_Active_Power") for ph in phases}
    imp_idx = {ph: name_to_idx.get(f"{ph}_Import_kWh")   for ph in phases}
    exp_idx = {ph: name_to_idx.get(f"{ph}_Export_kWh")   for ph in phases}

    # per-phase accumulators
    neg_count = {ph: 0 for ph in phases}
    sample_count = {ph: 0 for ph in phases}

    # trapezoidal integration state
    last_t = None
    last_p = {ph: None for ph in phases}
    imp_energy = {ph: 0.0 for ph in phases}  # kWh from +P
    exp_energy = {ph: 0.0 for ph in phases}  # kWh from -P

    # per-phase meter counters (first/last)
    imp_first = {ph: None for ph in phases}
    imp_last  = {ph: None for ph in phases}
    exp_first = {ph: None for ph in phases}
    exp_last  = {ph: None for ph in phases}

    # stream rows
    for row in reader:
        try:
            t = to_ms_from_num(row[ts_i])
        except Exception:
            continue
        if not (start_ms <= t < end_ms):
            continue

        # read counters (if present)
        for ph in phases:
            j = imp_idx.get(ph)
            if j is not None and j < len(row):
                try:
                    v = float(row[j])
                except Exception:
                    v = None
                if v is not None:
                    if imp_first[ph] is None:
                        imp_first[ph] = v
                    imp_last[ph] = v
            j = exp_idx.get(ph)
            if j is not None and j < len(row):
                try:
                    v = float(row[j])
                except Exception:
                    v = None
                if v is not None:
                    if exp_first[ph] is None:
                        exp_first[ph] = v
                    exp_last[ph] = v

        # power-based stats and integration
        for ph in phases:
            j = ap_idx.get(ph)
            p = None
            if j is not None and j < len(row):
                try:
                    p = float(row[j])
                except Exception:
                    p = None

            if p is not None and np.isfinite(p):
                sample_count[ph] += 1
                if p < 0:
                    neg_count[ph] += 1

                if last_t is not None and last_p[ph] is not None:
                    dt_h = (t - last_t) / 3600000.0  # hours
                    if dt_h > 0:
                        # trapezoid on positive/negative parts separately
                        p_prev = last_p[ph]
                        # import part (positive)
                        p1 = p_prev if p_prev > 0 else 0.0
                        p2 = p      if p      > 0 else 0.0
                        imp_energy[ph] += ((p1 + p2) * 0.5) * dt_h / 1000.0
                        # export part (negative)
                        n1 = -p_prev if p_prev < 0 else 0.0
                        n2 = -p      if p      < 0 else 0.0
                        exp_energy[ph] += ((n1 + n2) * 0.5) * dt_h / 1000.0

                last_p[ph] = p

        last_t = t

    # Print report
    print("\n[Diagnose Export Source (per phase)]")
    header_line = f"{'Phase':<4} {'NegP%':>7}  {'Imp_kWh(P)':>10} {'Exp_kWh(P)':>10}  {'Imp_kWh(m)':>10} {'Exp_kWh(m)':>10}   Verdict"
    print(header_line)
    print("-"*len(header_line))

    for ph in phases:
        neg_ratio = (100.0 * neg_count[ph] / sample_count[ph]) if sample_count[ph] > 0 else 0.0
        imp_p = imp_energy[ph]
        exp_p = exp_energy[ph]

        imp_m = float("nan")
        exp_m = float("nan")
        if imp_first[ph] is not None and imp_last[ph] is not None:
            imp_m = max(0.0, imp_last[ph] - imp_first[ph])
        if exp_first[ph] is not None and exp_last[ph] is not None:
            exp_m = max(0.0, exp_last[ph] - exp_first[ph])

        # Verdict heuristics
        verdict = "OK"
        if (exp_m == exp_m) and (exp_m > 0.5) and (neg_ratio > 90.0) and (not (imp_m == imp_m) or imp_m < 0.2):
            verdict = "CT likely reversed"
        elif (exp_m == exp_m) and exp_m > 0.3 and neg_ratio > 30.0:
            verdict = "Significant export on phase"
        elif (exp_m == exp_m) and exp_m > 0.05:
            verdict = "Small export"

        def _fmt(x):
            return f"{x:10.3f}" if (x == x) else f"{'':>10}"  # NaN-safe

        print(f"{ph:<4} {neg_ratio:7.2f}  {_fmt(imp_p)} {_fmt(exp_p)}  {_fmt(imp_m)} {_fmt(exp_m)}   {verdict}")

# -------- Main --------
def main():
    date_local = DATE_STR or today_local_str()
    tz_local = get_tz(LOCAL_TZ_NAME)
    print(f"Bucket={BUCKET} Prefix={PREFIX} Date(local)={date_local} TZ={LOCAL_TZ_NAME}")

    key = key_for_local_date_csv(date_local)
    print(f"[Key] Using aggregated file: s3://{BUCKET}/{key}")

    # Local day window -> UTC ms bounds
    start_local = datetime.fromisoformat(date_local).replace(tzinfo=tz_local)
    end_local = start_local + timedelta(days=1)
    start_ms = int(start_local.astimezone(timezone.utc).timestamp() * 1000)
    end_ms = int(end_local.astimezone(timezone.utc).timestamp() * 1000)

    # Compute hourly kWh from counters
    rows, header, raw_csv_text = hourly_kwh_from_counters(key, start_ms, end_ms)
    if not rows:
        print("No rows produced (no data in window?).")
        return

    # Save CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"hourly_kwh_{date_local}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[Write] Saved hourly kWh CSV: {out_path}")

    # Diagnose export sources (pure-Python, NumPy-array-free)
    try:
        diagnose_export_sources(raw_csv_text, start_ms, end_ms)
    except Exception as e:
        print(f"[Diagnose] Skipped: {e}")

if __name__ == "__main__":
    main()
