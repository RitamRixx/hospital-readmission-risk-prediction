import pandas as pd
from typing import Dict,Any,List


def transform_data(raw_records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not raw_records:
        print('no revords to transform')
        return pd.DataFrame()
    
    df = pd.DataFrame(raw_records)

    print("Transforming records")

    initial_count = len(df)
    df = df.drop_duplicates()

    duplicates_removed = initial_count - len(df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate records")

    print(f"Transformation complete: {len(df)} records ready")

    return df