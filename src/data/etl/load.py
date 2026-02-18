import json
from datetime import datetime
from typing import List, Dict
from sqlalchemy import text
from src.db.engine import get_engine

def load_data(records: List[Dict], replace_mode: bool = False) -> bool:
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

            archive_query = text("""
                INSERT INTO hospital_data_archive (payload, ingest_at)
                VALUES (:payload, :ingest_at)
            """)

            conn.execute(archive_query, values)
            print(f"Archived {len(records)} records to hospital_data_archive")

            if replace_mode:
                delete_query = text("""
                    DELETE FROM raw_hospital_data
                """)
                result = conn.execute(delete_query)
                print(f"Deleted {result.rowcount} existing records from raw_hospital_data")
            
            training_query = text("""
                INSERT INTO raw_hospital_data (payload, ingest_at)
                VALUES (:payload, :ingest_at)
            """)
            
            conn.execute(training_query, values)
            
            print(f"Successfully loaded {len(records)} records into raw_hospital_data")
            return True
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return False