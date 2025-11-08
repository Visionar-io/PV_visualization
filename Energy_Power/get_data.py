#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests, numpy as np, csv, io
from datetime import datetime, timezone

# ===== Your system =====
LAT, LON = 28.095, -15.487
PEAKPOWER_KW = 0.6
LOSS_PCT = 14.0
TILT_DEG = 26
AZIMUTH_DEG = 0                # PVGIS: 0=South, +90=West, -90=East
RADDATABASE = "PVGIS-SARAH3"   # SARAH3 available up to 2023 in v5.3
STARTYEAR = 2023
ENDYEAR   = 2023

# ===== Helpers =====
def parse_hourly_json_or_text(hourly_raw):
    """
    Return arrays: timestamps (str), P_watts (float).
    Accepts list-of-dicts or CSV/text blob.
    """
    # Nice JSON
    if isinstance(hourly_raw, list) and hourly_raw and isinstance(hourly_raw[0], dict):
        ts = [row["time"] for row in hourly_raw]
        if "P" not in hourly_raw[0]:
            raise RuntimeError("Hourly PV power 'P' missing; ensure pvcalculation=1 and PV params are set.")
        Pw = [float(row["P"]) for row in hourly_raw]
        return np.array(ts, dtype="U40"), np.array(Pw, dtype=float)

    # CSV/text
    blob = "\n".join(hourly_raw) if isinstance(hourly_raw, list) else str(hourly_raw)
    rows = None
    try:
        sample = "\n".join(blob.splitlines()[:5])
        dialect = csv.Sniffer().sniff(sample)
        rows = list(csv.reader(io.StringIO(blob), dialect))
    except Exception:
        for delim in [",", ";", "\t"]:
            try:
                rows = list(csv.reader(io.StringIO(blob), delimiter=delim)); break
            except Exception:
                pass
    if not rows:
        rows = [line.split() for line in blob.splitlines() if line.strip()]

    header = [h.strip() for h in rows[0]]
    norm = {h.lower().replace(" ", ""): i for i, h in enumerate(header)}
    def idx(name):
        key = name.lower().replace(" ", "")
        if key not in norm:
            raise KeyError(f"Missing column '{name}' in header {header}")
        return norm[key]

    i_time = idx("time")
    i_P = norm.get("p")
    if i_P is None:
        raise RuntimeError("Column 'P' (PV power) not found; ensure pvcalculation=1.")
    ts, Pw = [], []
    for r in rows[1:]:
        if not r or all(not str(x).strip() for x in r): continue
        ts.append(r[i_time]); Pw.append(float(r[i_P]))
    return np.array(ts, dtype="U40"), np.array(Pw, dtype=float)

def parse_any_time(ts: str) -> datetime:
    """
    Parse both ISO-8601 (e.g., '2023-01-01T00:00:00Z' or with offset)
    and compact 'YYYYMMDD:HHMM' returned by PVGIS in some modes.
    Returns a timezone-aware datetime when possible; otherwise naive.
    """
    s = ts.strip()
    # ISO variants first
    if "T" in s or "-" in s:
        s = s.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s)
        except Exception:
            pass
    # Compact forms: YYYYMMDD:HHMM or YYYYMMDD:HH
    # e.g., '20230101:0009' or '20230101:00'
    for fmt in ("%Y%m%d:%H%M", "%Y%m%d:%H"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # Last resort: try just date
    try:
        return datetime.strptime(s[:8], "%Y%m%d")
    except Exception as e:
        raise ValueError(f"Unrecognized time format: {ts}") from e

def datetimes_to_datestr(dt: datetime) -> str:
    return dt.date().isoformat()

# ===== Call PVGIS seriescalc =====
base_url = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
params = {
    "lat": LAT, "lon": LON,
    "startyear": STARTYEAR, "endyear": ENDYEAR,
    "raddatabase": RADDATABASE,
    "angle": TILT_DEG, "aspect": AZIMUTH_DEG,
    "mountingplace": "free",
    "pvtechchoice": "crystSi",
    "pvcalculation": 1,        # include PV power columns
    "peakpower": PEAKPOWER_KW,
    "loss": LOSS_PCT,
    "localtime": 1,            # local timestamps
    "timeformat": "iso8601",   # <-- ask for ISO timestamps to avoid compact format
    "outputformat": "json",
}

resp = requests.get(base_url, params=params, timeout=90)
try:
    resp.raise_for_status()
except requests.HTTPError as e:
    raise SystemExit(
        f"{e}\nHint: With SARAH3 in PVGIS v5.3, use years ≤ 2023. "
        f"If you need newer, switch to another DB when available."
    ) from None

payload = resp.json()
if "errors" in payload:
    raise RuntimeError(f"PVGIS error: {payload['errors']}")

hourly_raw = payload.get("outputs", {}).get("hourly")
if hourly_raw is None:
    raise RuntimeError("No 'hourly' data in response.")

# ---- Parse hourly and timestamps ----
timestamps, P_watts = parse_hourly_json_or_text(hourly_raw)

# Convert to datetimes (robust to both ISO and compact just in case)
dt = np.array([parse_any_time(t) for t in timestamps], dtype=object)

# Sort just in case
order = np.argsort(dt)
dt = dt[order]
P_watts = P_watts[order]

# ---- Compute per-sample duration in hours (handles DST days) ----
# Duration for sample i = (t[i+1]-t[i]) in hours; last sample gets same as previous
if len(dt) < 2:
    raise RuntimeError("Too few samples to compute durations.")
delta_hours = np.empty(len(dt), dtype=float)
delta_hours[:-1] = np.diff([d.timestamp() if d.tzinfo else d.replace(tzinfo=timezone.utc).timestamp() for d in dt]) / 3600.0
# For the last step, repeat the previous delta (typical for uniform series)
delta_hours[-1] = delta_hours[-2] if len(dt) > 1 else 1.0

# ---- Daily energy (kWh) = sum(P[W] * hours) / 1000 per day ----
dates = np.array([datetimes_to_datestr(d) for d in dt], dtype="U10")
uniq_dates, inv_idx = np.unique(dates, return_inverse=True)
daily_Wh = np.bincount(inv_idx, weights=P_watts * delta_hours)
daily_kWh = daily_Wh / 1000.0

# ---- Save CSVs ----
with open("pvgis_daily_energy.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["date", "energy_kWh"])
    for d, e in zip(uniq_dates, daily_kWh):
        w.writerow([d, f"{e:.6f}"])
print("Saved: pvgis_daily_energy.csv")

with open("pvgis_hourly.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["timestamp", "P_W"])
    for t, p in zip(timestamps, P_watts):
        w.writerow([t, f"{p:.3f}"])
print("Saved: pvgis_hourly.csv")

# ---- Console preview ----
print(f"\nSpan: {uniq_dates[0]} → {uniq_dates[-1]}  Days: {len(uniq_dates)}  Total: {daily_kWh.sum():.1f} kWh")
print("\nFirst 5 days:")
for d, e in list(zip(uniq_dates, daily_kWh))[:5]:
    print(f"  {d}: {e:.2f} kWh")
