import pandas as pd
import yaml
import os
from sqlalchemy import text
from src.db.engine import get_engine


def load_params(config_path: str = "params.yaml") -> dict:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"Error loading file params.yaml") from e
    
def ingest_data_postgres(output_path: str):

    params = load_params()

    view_name = params["database"]["view_name"]
    print(f"View: {view_name}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    engine = get_engine()


    query = text(f"""
        SELECT * FROM {view_name}
        ORDER BY id
    """)


    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

            if df.empty:
                raise ValueError("No data returned")
            
            print(f"Fetched {len(df)} records")
            print(f"Shape: {df.shape}")
            
            df.to_csv(output_path, index=False)


            return df
    except Exception as e:
        print(f"\nData ingestion failed: {e}")
        raise

def main():
    params = load_params()
    ingest_data_postgres(params["paths"]["raw_data"])

if __name__ == "__main__":
    main()
