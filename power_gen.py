# plot_avg_today_pvgis.py
# Average PV production for today's calendar date from a PVGIS timeseries CSV.
# Robust header/delimiter/power-column detection. No pandas required.

import csv, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import defaultdict

# --------- CONFIG ----------
CSV_FILE = "Timeseries_28.095_-15.487_SA3_1kWp_crystSi_14_27deg_2deg_2023_2023.csv"  # <-- set your file here
LOCAL_TZ = ZoneInfo("Atlantic/Canary")
MANUAL_POWER_COL_NAME = None  # e.g., "P (W)". Leave None to auto-detect.

# --------- HELPERS ----------
def sniff_delimiter(path: str) -> str:
    with open(path, "r", newline="") as f:
        sample = f.read(4096)
        try:
            return csv.Sniffer().sniff(sample, delimiters=";,").delimiter
        except Exception:
            return ";"  # PVGIS default

def parse_pvgis_time(tstr: str) -> datetime:
    tstr = tstr.strip()
    try:
        if tstr.endswith("Z"):
            return datetime.fromisoformat(tstr.replace("Z", "+00:00"))
        return datetime.fromisoformat(tstr)
    except Exception:
        return datetime.strptime(tstr, "%Y%m%d:%H%M").replace(tzinfo=timezone.utc)

def is_number(s: str) -> bool:
    try:
        float(s.replace(",", "."))
        return True
    except Exception:
        return False

def to_float(s: str) -> float:
    return float(s.replace(",", "."))

def detect_time_and_power_columns(header):
    # time col: any header containing 'time'
    time_idx = next((i for i,c in enumerate(header) if "time" in c.lower()), None)
    if time_idx is None:
        raise RuntimeError("No 'time' column found in header: " + " | ".join(header))

    # power col: common names
    lower = [c.strip().lower() for c in header]
    patterns = [
        r"^p$", r"^p\s*\(w\)$", r"^p\s*\[w\]$",
        r"^power", r"^p_?ac$", r"^p_?dc$", r"^pac$", r"^pdc$",
        r"^pmax", r"^pmax\s*\(w\)$"
    ]
    for i, col in enumerate(lower):
        if MANUAL_POWER_COL_NAME and col == MANUAL_POWER_COL_NAME.strip().lower():
            return time_idx, i
        for pat in patterns:
            if re.match(pat, col):
                return time_idx, i

    # not found -> return -1 to trigger numeric scoring
    return time_idx, -1

def choose_power_by_scoring(rows, time_idx, initial_power_idx):
    """
    If initial_power_idx == -1, scan rows and score all columns:
    - numeric_ratio: fraction of rows that parse as float
    - max_value: maximum numeric value (power columns usually have larger values)
    Prefer columns whose header contains 'p'/'power' as a tiebreaker.
    """
    header = rows[0]
    data = rows[1:]  # rest are data rows
    ncols = len(header)
    numeric_counts = np.zeros(ncols, dtype=float)
    max_vals = np.zeros(ncols, dtype=float)
    has_p_hint = np.array([("p" in header[i].lower() or "power" in header[i].lower()) for i in range(ncols)])

    # sample up to first 5000 data rows to score
    sample_rows = data[:5000] if len(data) > 5000 else data

    for row in sample_rows:
        if len(row) != ncols:
            continue
        for i in range(ncols):
            if i == time_idx:
                continue
            val = row[i].strip()
            if is_number(val):
                numeric_counts[i] += 1
                fv = to_float(val)
                if fv > max_vals[i]:
                    max_vals[i] = fv

    total = max(1, len(sample_rows))
    numeric_ratio = numeric_counts / total

    # mask impossible columns
    numeric_ratio[time_idx] = -1  # never pick time

    # build a composite score: prioritize numeric_ratio, then max_vals (scaled), then hint
    # normalize max_vals
    if max_vals.max() > 0:
        max_norm = max_vals / max_vals.max()
    else:
        max_norm = max_vals
    score = numeric_ratio + 0.2 * max_norm + 0.05 * has_p_hint.astype(float)

    # require at least 70% numeric rows to be safe
    score[numeric_ratio < 0.7] = -1

    best_idx = int(np.argmax(score))
    if score[best_idx] <= 0:
        raise RuntimeError("Couldn't detect a numeric power column after scoring.")
    return best_idx

def load_time_power(csv_path):
    delim = sniff_delimiter(csv_path)

    # Read all rows after finding the header (first row containing 'time')
    header = None
    rows = []
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f, delimiter=delim)
        for row in r:
            if not row:
                continue
            if header is None:
                if any("time" in c.lower() for c in row):
                    header = [c.strip() for c in row]
                    rows.append(header)
            else:
                # ensure consistent column count by padding/trimming
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                elif len(row) > len(header):
                    row = row[:len(header)]
                rows.append(row)

    if not rows:
        raise RuntimeError("No header/data found. Check the file path and delimiter.")

    time_idx, power_idx = detect_time_and_power_columns(rows[0])
    if power_idx == -1:
        power_idx = choose_power_by_scoring(rows, time_idx, power_idx)

    # Parse
    times_utc, power_W = [], []
    for row in rows[1:]:
        t_str = row[time_idx].strip()
        p_str = row[power_idx].strip()
        if not t_str or not p_str or not is_number(p_str):
            continue
        try:
            dt = parse_pvgis_time(t_str).astimezone(timezone.utc)
        except Exception:
            continue
        times_utc.append(dt)
        power_W.append(to_float(p_str))

    if not times_utc:
        raise RuntimeError("Parsed header but got 0 data rows. Are the date/number formats unusual?")
    return np.array(times_utc), np.array(power_W, dtype=float), rows[0], time_idx, power_idx

# --------- MAIN ----------
def main():
    times_utc, power, header, t_idx, p_idx = load_time_power(CSV_FILE)
    print(f"[INFO] Header: {header}")
    print(f"[INFO] Using time column: '{header[t_idx]}'  power column: '{header[p_idx]}'")
    print(f"[INFO] Loaded {len(times_utc)} samples: {times_utc[0]} … {times_utc[-1]} (UTC)")

    # Pick today's local calendar date
    today_local = datetime.now(LOCAL_TZ).date()
    mm, dd = today_local.month, today_local.day

    # Select today's LOCAL date across all years present
    sel = []
    for t_utc, p in zip(times_utc, power):
        t_loc = t_utc.astimezone(LOCAL_TZ)
        if t_loc.month == mm and t_loc.day == dd:
            sel.append((t_loc, p))

    # Fallback: if nothing (rare with time-zone boundaries), try UTC calendar day
    if not sel:
        print("[WARN] No local-time matches for today; trying UTC day.")
        for t_utc, p in zip(times_utc, power):
            if t_utc.month == mm and t_utc.day == dd:
                sel.append((t_utc.astimezone(LOCAL_TZ), p))

    if not sel:
        raise RuntimeError(f"No data for {today_local:%m-%d} in file (local or UTC).")

    # Group by time-of-day (HH:MM), average over years
    buckets = defaultdict(list)
    for t_loc, p in sel:
        buckets[(t_loc.hour, t_loc.minute)].append(p)

    hm_sorted = sorted(buckets.keys())
    avgP = np.array([np.mean(buckets[hm]) for hm in hm_sorted])

    # Build x-axis as today's date with those times
    x_dt = [
        datetime.combine(today_local, datetime.min.time(), tzinfo=LOCAL_TZ).replace(hour=h, minute=m)
        for (h, m) in hm_sorted
    ]
    x_num = mdates.date2num(x_dt)

    # Energy (kWh) via trapezoid integration
    x_sec = (x_num - x_num[0]) * 86400.0
    energy_kWh = (np.trapz(avgP, x_sec) / 3600.0) / 1000.0

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot_date(x_num, avgP/1000.0, "-", linewidth=2)
    plt.title(f"Average PV Production on {today_local.strftime('%b %d')} (from file)")
    plt.xlabel(f"Time ({LOCAL_TZ.key})")
    plt.ylabel("Average Power [kW]")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LOCAL_TZ))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pvgis_avg_today.png", dpi=150)
    plt.show()

    print(f"[RESULT] Average daily energy for {today_local:%b %d}: {energy_kWh:.2f} kWh")
    print("Saved figure: pvgis_avg_today.png")

if __name__ == "__main__":
    main()
