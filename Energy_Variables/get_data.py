import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import numpy as np
import csv

class Aws:
    def __init__(self, bucket, prefix, local_time):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.local_time = local_time
        self.s3 = self._make_s3_client()

    def _make_s3_client(self):
        """Create a low-level S3 client without requiring an AWS profile."""
        region = "us-west-2"  # default fallback
        try:
            session = boto3.Session()
            probe = session.client("s3")
            loc = probe.get_bucket_location(Bucket=self.bucket)["LocationConstraint"]
            if loc:
                region = loc
        except ClientError:
            pass

        cfg = Config(
            region_name=region,
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=5,
            read_timeout=30,
            max_pool_connections=64,
        )
        print(f"[Aws] Using region={region!r}")
        return boto3.client("s3", config=cfg)

    def s3_object(self, name):
        """
        Download an object from S3 and return its text content.
        - name: file name (e.g., '2025/10/31.csv' or 'mydata.csv')
        Returns: decoded UTF-8 string.
        """
        key = self.prefix + name.lstrip("/")
        print(f"[Aws] Downloading s3://{self.bucket}/{key}")
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            data = obj["Body"].read().decode("utf-8", errors="replace")
            print(f"[Aws] Downloaded {len(data)} bytes.")
            return data
        except ClientError as e:
            print(f"[Aws] Error fetching {key}: {e}")
            return None 

class Load:
    def __init__(self, start_date, stop_date, number_phases, aws=Aws):
        self.start_date = start_date
        self.stop_date = stop_date
        self.number_phases = number_phases
        self.aws = aws

    def download_data(self):
        energy_data = []
        power_data = []
        energy_header = []
        power_header = []

        if self.number_phases == 1:
            pass  # (Not implemented yet)

        elif self.number_phases == 3:
            energy_columns = [
                "L1_Import_kWh","L2_Import_kWh","L3_Import_kWh",
                "L1_Export_kWh","L2_Export_kWh","L3_Export_kWh",
                "Total_Active_Energy","Total_Reactive_Energy"
            ]
            power_columns = [
                "L1_Active_Power","L2_Active_Power","L3_Active_Power",
                "Total_System_Power","Total_System_Reactive_Power","Total_System_Apparent_Power"
            ]

            dates = get_dates_between(self.start_date, self.stop_date)
            for date_str in dates:
                y, m, d = date_str[0:4], date_str[5:7], date_str[8:10]
                key_name = f"{y}/{m}/{d}.csv"

                text = self.aws.s3_object(key_name)
                if not text:
                    continue

                lines = [ln for ln in text.splitlines() if ln.strip()]
                if not lines:
                    continue

                reader = csv.reader(lines)
                header = next(reader, None)
                if header is None:
                    continue

                name_to_idx = {n: i for i, n in enumerate(header)}
                ts_i = name_to_idx.get("ts")

                energy_cols_present = [c for c in energy_columns if c in name_to_idx]
                power_cols_present  = [c for c in power_columns  if c in name_to_idx]

                if not energy_cols_present and not power_cols_present:
                    continue

                # Build headers once (include timestamp first)
                if not energy_header and energy_cols_present:
                    energy_header = ["ts"] + energy_cols_present
                if not power_header and power_cols_present:
                    power_header = ["ts"] + power_cols_present

                # Process each row
                for row in reader:
                    if ts_i is None or ts_i >= len(row):
                        continue
                    ts_val = _to_float(row[ts_i])

                    if energy_cols_present:
                        vals = [_to_float(row[name_to_idx[c]]) for c in energy_cols_present]
                        energy_data.append([ts_val] + vals)

                    if power_cols_present:
                        vals = [_to_float(row[name_to_idx[c]]) for c in power_cols_present]
                        power_data.append([ts_val] + vals)

        # Convert to NumPy arrays
        energy_array = np.array(energy_data, dtype=float) if energy_data else np.empty((0, 0))
        power_array  = np.array(power_data,  dtype=float) if power_data  else np.empty((0, 0))

        # Wrap results in clear dictionaries
        energy = {"header": energy_header, "data": energy_array}
        power  = {"header": power_header,  "data": power_array}

        return energy, power

    def compute_time_span(self, time_span: int):
            """
            Aggregate:
            - ENERGY (cumulative counters): per-bin kWh = last - first (clipped at 0)
            - POWER  (instantaneous): per-bin average
            Returns (two dicts with np arrays of dtype=object so date stays string):
            energy_res = {"header": ["date","bin","start_ts","end_ts", ...], "data": ndarray(object)}
            power_res  = {"header": ["date","bin","start_ts","end_ts", ...], "data": ndarray(object)}
            """
            energy, power = self.download_data()
            tz_local = _get_tz(self.aws.local_time)

            # ===== ENERGY =====
            energy_res = {"header": [], "data": np.empty((0, 0), dtype=object)}
            e_hdr = energy.get("header") or []
            E = energy.get("data")
            if isinstance(E, np.ndarray) and E.size > 0 and len(e_hdr) >= 2:
                # E columns: [ts, e1, e2, ...]
                e_ts = np.asarray(E[:, 0], dtype=np.int64)
                e_cols = e_hdr[1:]
                Evals = np.asarray(E[:, 1:], dtype=float)

                date_to_idx = _group_by_local_date(e_ts, tz_local)
                out_rows = []
                out_header = ["date", "bin", "start_ts", "end_ts"] + e_cols

                for date_str, idxs in sorted(date_to_idx.items()):
                    idxs = np.array(sorted(idxs))
                    bins = _day_bins(date_str, tz_local, time_span)

                    for bin_idx, b_start_ms, b_end_ms in bins:
                        mask = (e_ts[idxs] >= b_start_ms) & (e_ts[idxs] < b_end_ms)
                        if not np.any(mask):
                            row_vals = [np.nan] * len(e_cols)
                            out_rows.append([date_str, bin_idx, b_start_ms, b_end_ms] + row_vals)
                            continue

                        sub_idx = idxs[mask]
                        vals = Evals[sub_idx, :]  # shape (m, k)

                        # Per-column first/last finite and delta
                        deltas = []
                        for j in range(vals.shape[1]):
                            col = vals[:, j]
                            finite = np.isfinite(col)
                            if not np.any(finite):
                                deltas.append(np.nan)
                                continue
                            first_v = col[finite][0]
                            last_v  = col[finite][-1]
                            d = last_v - first_v
                            if np.isfinite(d) and d < 0:
                                d = 0.0
                            deltas.append(d if np.isfinite(d) else np.nan)

                        out_rows.append([date_str, bin_idx, b_start_ms, b_end_ms] + deltas)

                if out_rows:
                    energy_res["header"] = out_header
                    energy_res["data"] = np.array(out_rows, dtype=object)

            # ===== POWER =====
            power_res = {"header": [], "data": np.empty((0, 0), dtype=object)}
            p_hdr = power.get("header") or []
            P = power.get("data")
            if isinstance(P, np.ndarray) and P.size > 0 and len(p_hdr) >= 2:
                # P columns: [ts, p1, p2, ...]
                p_ts = np.asarray(P[:, 0], dtype=np.int64)
                p_cols = p_hdr[1:]
                Pvals = np.asarray(P[:, 1:], dtype=float)

                date_to_idx = _group_by_local_date(p_ts, tz_local)
                out_rows = []
                out_header = ["date", "bin", "start_ts", "end_ts"] + p_cols

                for date_str, idxs in sorted(date_to_idx.items()):
                    idxs = np.array(sorted(idxs))
                    bins = _day_bins(date_str, tz_local, time_span)

                    for bin_idx, b_start_ms, b_end_ms in bins:
                        mask = (p_ts[idxs] >= b_start_ms) & (p_ts[idxs] < b_end_ms)
                        if not np.any(mask):
                            row_vals = [np.nan] * len(p_cols)
                            out_rows.append([date_str, bin_idx, b_start_ms, b_end_ms] + row_vals)
                            continue

                        sub_idx = idxs[mask]
                        vals = Pvals[sub_idx, :]
                        with np.errstate(all='ignore'):
                            mean_vals = np.nanmean(vals, axis=0)
                        out_rows.append([date_str, bin_idx, b_start_ms, b_end_ms] + mean_vals.tolist())

                if out_rows:
                    power_res["header"] = out_header
                    power_res["data"] = np.array(out_rows, dtype=object)

            return energy_res, power_res

class EnergyCosts():
    def __init__(self):
        pass
     
from datetime import datetime, timedelta, timezone

def get_dates_between(start_date: str, end_date: str):
    """
    Devuelve una lista de fechas (YYYY-MM-DD) entre start_date y end_date (incluidas).
    Acepta formato 'AAAA-MM-DD'. Si start_date > end_date, las intercambia.
    """
    fmt = "%Y-%m-%d"
    d0 = datetime.strptime(start_date, fmt).date()
    d1 = datetime.strptime(end_date, fmt).date()
    if d0 > d1:
        d0, d1 = d1, d0
    days = (d1 - d0).days
    return [(d0 + timedelta(days=i)).strftime(fmt) for i in range(days + 1)]

def _to_ms_from_num(x):
    t = int(float(x))
    return t if t >= 10**12 else t * 1000

def _get_tz(tz_name):
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz_name)
    except Exception:
        from dateutil import tz
        return tz.gettz(tz_name)

def _group_by_local_date(ts_ms_arr, tz_local):
    groups = {}
    for i, tms in enumerate(ts_ms_arr):
        dt_local = datetime.fromtimestamp(tms/1000.0, tz=timezone.utc).astimezone(tz_local)
        key = dt_local.strftime("%Y-%m-%d")
        groups.setdefault(key, []).append(i)
    return groups

def _date_to_int(date_str):
    """Convert 'YYYY-MM-DD' → int YYYYMMDD"""
    return int(date_str.replace("-", ""))

def _day_bins(local_date_str, tz_local, span_sec):
    y, m, d = map(int, local_date_str.split("-"))
    start_local = datetime(y, m, d, 0, 0, 0, tzinfo=tz_local)
    end_local = start_local + timedelta(days=1)
    bins = []
    cur = start_local
    k = 0
    while cur < end_local:
        nxt = min(cur + timedelta(seconds=span_sec), end_local)
        bins.append((
            k,
            int(cur.astimezone(timezone.utc).timestamp() * 1000),
            int(nxt.astimezone(timezone.utc).timestamp() * 1000)
        ))
        cur = nxt
        k += 1
    return bins

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan
    
if __name__ == "__main__":
    BUCKET = "pm-metering"
    KEY = "Carmelo" 
    LOCAL_TIME = "Atlantic/Canary"
    aws = Aws(BUCKET, KEY, LOCAL_TIME)
    START_DATE = "2025-10-30"
    STOP_DATE = "2025-10-30"
    NUMBER_PHASES = 3
    load = Load (START_DATE, STOP_DATE, NUMBER_PHASES, aws)
    print(load.compute_time_span(3600))