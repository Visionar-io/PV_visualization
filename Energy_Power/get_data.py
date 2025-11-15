#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import requests, numpy as np, csv, io
from datetime import datetime, date, timezone

# ===== Your system =====
LAT, LON = 28.095, -15.487
PEAKPOWER_KW = 0.6
LOSS_PCT = 14.0
TILT_DEG = 26
AZIMUTH_DEG = 0                # PVGIS: 0=South, +90=West, -90=East
RADDATABASE = "PVGIS-SARAH3"   # SARAH3 available up to 2023 in v5.3

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
        if not r or all(not str(x).strip() for x in r):
            continue
        ts.append(r[i_time]); Pw.append(float(r[i_P]))
    return np.array(ts, dtype="U40"), np.array(Pw, dtype=float)

def parse_any_time(ts: str) -> datetime:
    """
    Parse both ISO-8601 and compact 'YYYYMMDD:HHMM' timestamps.
    Returns a timezone-aware datetime in UTC.
    """
    s = ts.strip()
    # ISO variants first
    if "T" in s or "-" in s:
        # If PVGIS returns "Z", keep it as UTC
        s = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
            # Ensure UTC tzinfo
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            pass
    # Compact forms
    for fmt in ("%Y%m%d:%H%M", "%Y%m%d:%H"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    # Last resort: just date
    try:
        dt = datetime.strptime(s[:8], "%Y%m%d")
        return dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        raise ValueError(f"Unrecognized time format: {ts}") from e

def build_second_resolution_model(dt_array, P_kW_array, target_date: date):
    """
    Build an interpolating function P_of_second(t) based on all samples
    from 'target_date' in dt_array / P_kW_array.

    All times are handled in UTC.
    """
    # target_date is interpreted as UTC date
    mask = np.array([d.astimezone(timezone.utc).date() == target_date for d in dt_array], dtype=bool)
    if not mask.any():
        raise ValueError(f"No data found for date {target_date}")

    day_dt = dt_array[mask]
    day_P = P_kW_array[mask]

    # Convert timestamps to "seconds since midnight UTC"
    sec = np.array(
        [
            d.astimezone(timezone.utc).hour * 3600
            + d.astimezone(timezone.utc).minute * 60
            + d.astimezone(timezone.utc).second
            for d in day_dt
        ],
        dtype=float
    )

    # Sort by time of day
    order = np.argsort(sec)
    sec = sec[order]
    day_P = day_P[order]

    day_length = 24 * 3600.0

    # Ensure bounds [0, 24h] with P=0 outside the measured daylight window
    if sec[0] > 0.0:
        sec = np.insert(sec, 0, 0.0)
        day_P = np.insert(day_P, 0, 0.0)

    if sec[-1] < day_length:
        sec = np.append(sec, day_length)
        day_P = np.append(day_P, 0.0)

    def P_of_second(t):
        """
        Power model function in UTC.
        - t can be:
            * seconds since midnight UTC (float/int), or
            * a datetime (naive -> treated as UTC, aware -> converted to UTC).
        Returns power in kW at that instant.
        """
        from datetime import datetime as _dt

        if isinstance(t, _dt):
            # If timezone-aware, convert to UTC; else assume it's UTC
            if t.tzinfo is not None:
                t_utc = t.astimezone(timezone.utc)
            else:
                t_utc = t.replace(tzinfo=timezone.utc)
            t = (
                t_utc.hour * 3600
                + t_utc.minute * 60
                + t_utc.second
                + t_utc.microsecond / 1e6
            )

        t = float(t)
        t = t % day_length
        return float(np.interp(t, sec, day_P))

    return P_of_second

# ===== Call PVGIS and build model =====
from datetime import date

def get_model(start_date: str,
              end_date: str,
              peakpower_kw: float = PEAKPOWER_KW,
              target_date: str | None = None):
    """
    Call PVGIS, get hourly PV power for [start_date, end_date],
    and build a per-second model for `target_date` (defaults to start_date,
    all interpreted in UTC).

    Dates are strings 'YYYY-MM-DD', year is forced to 2023 (SARAH3).
    """
    # ---- Parse incoming strings into date objects ----
    user_start = date.fromisoformat(start_date)
    user_end   = date.fromisoformat(end_date)

    # Force year to 2023, keep same month and day
    start_date_obj = user_start.replace(year=2023)
    end_date_obj   = user_end.replace(year=2023)

    if end_date_obj < start_date_obj:
        raise ValueError("end_date must be >= start_date")

    if target_date is None:
        target_date_obj = start_date_obj
    else:
        # Only change the year if you also want target_date clamped to 2023
        td = date.fromisoformat(target_date)
        target_date_obj = td.replace(year=2023)

    start_year = start_date_obj.year
    end_year   = end_date_obj.year

    base_url = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
    params = {
        "lat": LAT, "lon": LON,
        "startyear": start_year, "endyear": end_year,
        "raddatabase": RADDATABASE,
        "angle": TILT_DEG, "aspect": AZIMUTH_DEG,
        "mountingplace": "free",
        "pvtechchoice": "crystSi",
        "pvcalculation": 1,
        "peakpower": peakpower_kw,
        "loss": LOSS_PCT,
        # <<<<<< IMPORTANT: ask PVGIS for UTC times
        "localtime": 0,
        "timeformat": "iso8601",
        "outputformat": "json",
    }

    resp = requests.get(base_url, params=params, timeout=90)
    resp.raise_for_status()
    payload = resp.json()
    if "errors" in payload:
        raise RuntimeError(f"PVGIS error: {payload['errors']}")

    hourly_raw = payload.get("outputs", {}).get("hourly")
    if hourly_raw is None:
        raise RuntimeError("No 'hourly' data in response.")

    # ---- Parse hourly and timestamps (as UTC) ----
    timestamps, P_watts = parse_hourly_json_or_text(hourly_raw)
    dt = np.array([parse_any_time(t) for t in timestamps], dtype=object)

    # Sort just in case
    order = np.argsort(dt)
    dt = dt[order]
    P_watts = P_watts[order]
    timestamps = timestamps[order]

    # ---- Filter to requested UTC date range ----
    mask = np.array(
        [(d.astimezone(timezone.utc).date() >= start_date_obj) and
         (d.astimezone(timezone.utc).date() <= end_date_obj)
         for d in dt],
        dtype=bool
    )
    dt = dt[mask]
    P_watts = P_watts[mask]
    timestamps = timestamps[mask]

    if len(dt) == 0:
        raise RuntimeError("No data found in the requested date range.")

    # ---- Convert to kW ----
    P_kW = P_watts / 1000.0

    # Optional: save hourly CSV (timestamps are UTC)
    out_name = "pvgis_hourly_kw.csv"
    with open(out_name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_UTC", "P_kW"])
        for t, p in zip(timestamps, P_kW):
            w.writerow([t, f"{p:.6f}"])
    print(f"Saved: {out_name}")
    print(f"Span (UTC dates): {start_date_obj.isoformat()} → {end_date_obj.isoformat()}  Samples: {len(P_kW)}")

    # ---- Build per-second model for target_date_obj (UTC date) ----
    P_of_second = build_second_resolution_model(dt, P_kW, target_date_obj)

    return P_of_second, dt, P_kW


# ===== CLI entrypoint (optional) =====
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            f"Usage: {sys.argv[0]} START_DATE END_DATE\n"
            f"Example: {sys.argv[0]} 2023-01-01 2023-01-31"
        )

    # Here CLI args are strings, as expected by get_model
    START_DATE = sys.argv[1]
    END_DATE   = sys.argv[2]

    model, dt_all, PkW_all = get_model(START_DATE, END_DATE)

    # Quick demo: 06:30, 12:45, 18:30 UTC on the (clamped) START_DATE
    for h, m in [(6, 30), (12, 45), (18, 30)]:
        sec = h * 3600 + m * 60
        print(f"{h:02d}:{m:02d}:00 UTC -> {model(sec):.3f} kW")
