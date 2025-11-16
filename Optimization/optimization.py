import os
import sys
from datetime import date, datetime, timedelta

# Add project root (PV_visualization) to sys.path
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, PROJECT_ROOT)

from Energy_Consumption.kwh import get_data_energy_consumption
from Energy_Consumption.get_data import Load, Aws, EnergyCosts
from Energy_Prices.get_prices import extract_day_hour_pcb_from_url
from Energy_Power.get_data import get_model
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_date_range(start_date: str, end_date: str):
    """
    Given start_date and end_date as 'YYYY-MM-DD',
    return a list of date strings for all days between them (inclusive).
    """
    # Parse strings to date objects
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    if end < start:
        raise ValueError("end_date must be on or after start_date")

    days = []
    current = start
    while current <= end:
        days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return days


if __name__ == "__main__":
    # VARIABLES
    ## AWS Variables
    BUCKET = "pm-metering"
    KEY = "Carmelo"
    ## Date variables
    START_DATE = "2025-10-29"
    STOP_DATE = "2025-10-30" 
    LOCAL_TIME = "Atlantic/Canary"
    ## Electrical Variables
    NUMBER_PHASES = 3
    ## Panels kW
    number_panels = 2
    Panels_kw = 0.6 * number_panels
    logging.info(f"Panels capacity {Panels_kw} kw")
    ## Batteries kW
    number_batteries = 2
    Batteries_kw = 260 * 12 * number_batteries / 1e3
    logging.info(f"Panels capacity {Batteries_kw} kw")
    ## Inverter kw
    Inverter_kw = 5
    logging.info(f"Inverter capacity {Inverter_kw} kw")
    # -----------------------------*_-------------------------
    # Get Dates
    dates = get_date_range(START_DATE, STOP_DATE)
    logging.info(f"Dates: {dates}")
    # -----------------------------*_-------------------------
    for date_item in dates:
        logging.info(f"current date: {date_item}")
        # PV Generation
        P_of_second, dt, P_kW = get_model(
        start_date=date_item,
        peakpower_kw=Panels_kw,
        )

        # Example: power at a specific UTC datetime
        from datetime import datetime, timezone
        t_utc = datetime(2025, 10, 29, 6, 45, tzinfo=timezone.utc)
        print("P(2023-10-29 12:45 UTC) =", P_of_second(t_utc), "kW")

        # -----------------------------*_-------------------------
        # Energy Consumption
        aws = Aws(BUCKET, KEY, LOCAL_TIME)
        load = Load (date_item, NUMBER_PHASES, aws)
        energy = EnergyCosts(date_item)
        energy, power = load.download_data()
        print("Energy header:", energy["header"])
        print("Energy shape:", energy["data"].shape)
        print("First 5 energy rows:\n", energy["data"][:5])

        print("Power header:", power["header"])
        print("Power shape:", power["data"].shape)
        print("First 5 power rows:\n", power["data"][:5])
        # -----------------------------*_-------------------------
        # Energy Costs
        hc = extract_day_hour_pcb_from_url (date_item)
        print(hc)

# import logging
# import sys
# sys.path.insert(0, "..")
# from Energy_Consumption.kwh import get_data_energy_consumption
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# #Variables
# # "YYYY-MM-DD" format
# START_DATE = "01-10-2025"
# END_DATE = "02-10-2025"

# # -----------------------_*-------------------------------
# # Get all the dates between START_DATE and END_DATE

# # Get electrical consumption data
# dt, P_kW = get_data_energy_consumption(START_DATE, END_DATE)
# # Get hourly cost data
# # get power generation data

# # 


