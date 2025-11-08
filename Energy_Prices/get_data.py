import json
import requests

def extract_day_hour_pcb_from_url(url):
    # Download the JSON file
    response = requests.get(url, timeout=30)
    response.raise_for_status()  # stop if there's an error
    data = response.json()       # parse JSON directly

    # Depending on structure: try to find the list
    if isinstance(data, dict):
        if "PVPC" in data:
            records = data["PVPC"]
        elif "data" in data:
            records = data["data"]
        else:
            # fallback: first list found in the dict
            records = next((v for v in data.values() if isinstance(v, list)), [])
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Unknown JSON structure")

    # Extract Dia, Hora, PCB
    result = [(r.get("Dia"), r.get("Hora"), r.get("PCB")) 
              for r in records if r.get("Dia") and r.get("Hora") and "PCB" in r]

    print("Dia,Hora,PCB")
    for dia, hora, pcb in result:
        print(f"{dia},{hora},{pcb}")

# Example URL (replace with your actual JSON URL)
url = "http://api.esios.ree.es/archives/70/download?date=2025-10-20"
extract_day_hour_pcb_from_url(url)
