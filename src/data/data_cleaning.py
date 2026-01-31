import pandas as pd
import numpy as np
import yaml
import os

def load_params(config_path: str = "params.yaml") -> dict:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"Error loading file params.yaml") from e
    

def clean_data(input_path: str, output_path: str):
    params = load_params()

    remove_duplicates = params["data_cleaning"]["remove_duplicates"]

    df = pd.read_csv(input_path)
    print(f"Initial shape: {df.shape}")


    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows")


    print(f"Final shape: {df.shape}")
    print(f"columns:{df.columns}")


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

def main():
        
    params = load_params()
    clean_data(
        params["paths"]["raw_data"],
        params["paths"]["cleaned_data"]
    )

if __name__ == "__main__":
    main()

