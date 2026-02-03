import pandas as pd
import numpy as np
import yaml
import os
import json


def load_params():
    try: 
        with open('params.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise Exception(f'Error loading file params.yaml') from e
    

def map_age(age_str, age_map):
    return age_map.get(age_str, None)

def icd9_group(code, ranges, default="Other", unknown="Unknown"):
    if pd.isna(code) or code == "?":
        return unknown
    
    try:
        code = float(code)
    except ValueError:
        return unknown

    for group, (low, high) in ranges.items():
        if low <= code < high:
            return group
       
    return default


def feature_engineering(input_path: str, output_path: str):
    params = load_params()
    params_fe = params["feature_engineering"]

    df = pd.read_csv(input_path)
    print(f"Input shape: {df.shape}")
    print(f"columns: {df.columns}")

    df["age"] = df["age"].apply(
        lambda x: map_age(x, params_fe["age_mapping"])
    )

    icd_ranges = params_fe["icd9_mapping"]["ranges"]
    default_grp = params_fe["icd9_mapping"]["default"]
    unknown_grp = params_fe["icd9_mapping"]["unknown"]
    handle_missing = params["data_cleaning"]["handle_missing"]

    # df["diag_1_group"] = df["diag_1"].apply(
    #     lambda x: icd9_group(x, icd_ranges, default_grp, unknown_grp)
    # )

    for diag_col in params_fe["diagnosis_columns"]:
        df[f"{diag_col}_group"] = df[diag_col].apply(
            lambda x: icd9_group(x, icd_ranges, default_grp, unknown_grp)
        )

    if params_fe["rare_category_grouping"]["enabled"]:
        threshold = params_fe["rare_category_grouping"]["threshold"]
        rare_cols = params_fe["rare_category_grouping"]["columns"]
        group_name = params_fe["rare_category_grouping"]["group_name"]
    
        for col in rare_cols:
            if col in df.columns:
                value_counts = df[col].value_counts(normalize=True)
                keep_values = value_counts[value_counts >= threshold].index
                df[col] = df[col].astype(str)
                df[col] = df[col].where(df[col].isin(keep_values.astype(str)), group_name)
                print(f"Grouped rare categories in {col}")


    meds = params_fe["medication_columns"]

    df["num_med_changes"] = df[meds].isin(["Up", "Down"]).sum(axis=1)
    df["num_med_active"] = (df[meds] != "No").sum(axis=1)


    df["total_visits"] = (
        df["number_outpatient"]
        + df["number_emergency"]
        + df["number_inpatient"]
    )

    df["interaction_visits_meds"] = (
        df["total_visits"] * df["num_med_active"]
    )


    insulin_map = params_fe["insulin_mapping"]
    df["insulin_coded"] = df["insulin"].map(insulin_map)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            if handle_missing["numeric"] == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif handle_missing["numeric"] == "mean":
                df[col].fillna(df[col].mean(), inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            if handle_missing["categorical"] == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
    


    drop_cols = params_fe["feature_selection"]["drop_column"]
    df = df.drop(columns=drop_cols, errors="ignore")


    print(f"Output shape: {df.shape}")


    feature_info = {
        'all_columns': df.columns.tolist(),
    }

    print(f"null values {df.isnull().sum().sum()}")
   

    os.makedirs('models', exist_ok=True)
    with open('models/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Featured data saved to: {output_path}")


def main():
    params = load_params()
    feature_engineering(
        params["paths"]["cleaned_data"],
        params["paths"]["featured_data"]
    )


if __name__ == "__main__":
    main()
