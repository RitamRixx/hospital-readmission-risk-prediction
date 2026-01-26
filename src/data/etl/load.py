import json
from datetime import datetime
from typing import List, Dict
from sqlalchemy import text
from src.db.engine import get_engine

def load_data(records: List[Dict]) -> bool:
    if not records:
        print("no record to load")
        return False
    
    engine = get_engine()

    try:
        with engine.begin() as conn:
            timestamp = datetime.utcnow()
            values = [
                {
                    'payload': json.dumps(record),
                    'ingest_at': timestamp
                }
                for record in records
            ]
            
            query = text("""
                INSERT INTO raw_hospital_data (payload, ingest_at)
                VALUES (:payload, :ingest_at)
            """)
            
            conn.execute(query, values)
            
            print(f"Successfully loaded {len(records)} records into raw_hospital_data")
            return True
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return False