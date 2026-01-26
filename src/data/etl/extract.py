import json
import os
import requests
from typing import Dict,Any,List,Tuple

STATE_FILE = "data/etl_state/state.json"

def load_state() -> int:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
            return data.get("last_offset", 0)
    return 0

def save_state(offset: int) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

    with open(STATE_FILE, 'w') as f:
        json.dump({'last_offset': offset}, f, indent=2)

def extract_data(api_url: str, batch_size: int = 100) -> Tuple[List[Dict], int]:
    current_offset = load_state()

    params = {
        'offset': current_offset,
        'limit': batch_size
    }

    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        records = data if isinstance(data, list) else []

        if not records:
            print("No new record found")
        else:
            print(f"Fetched {len(records)} records starting at {current_offset}")

        return records, current_offset
    except requests.exceptions.RequestException as e:
        print(f"Error extracting data: {e}")
        raise e