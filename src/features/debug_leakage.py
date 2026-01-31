import pandas as pd
import yaml

# Load params
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

target_col = params["feature_engineering"]["target"]["column_name"]

print("="*60)
print("DEBUGGING DATA LEAKAGE")
print("="*60)

# Check featured_data.csv (after feature_engineering)
print("\n1. Checking featured_data.csv:")
featured = pd.read_csv('data/processed/featured_data.csv')
print(f"   Shape: {featured.shape}")
print(f"   Columns: {featured.columns.tolist()}")
print(f"   Has 'readmitted': {'readmitted' in featured.columns}")
print(f"   Has '{target_col}': {target_col in featured.columns}")

if target_col in featured.columns:
    print(f"   Target distribution: {featured[target_col].value_counts().to_dict()}")

# Check train.csv (after data_split)
print("\n2. Checking train.csv:")
train = pd.read_csv('data/final/train.csv')
print(f"   Shape: {train.shape}")
print(f"   Columns: {train.columns.tolist()}")
print(f"   Has '{target_col}': {target_col in train.columns}")

if target_col in train.columns:
    print(f"   Target distribution: {train[target_col].value_counts().to_dict()}")

# Check if target is in BOTH X and y (LEAKAGE!)
print("\n3. LEAKAGE CHECK:")
X_cols = [col for col in train.columns if col != target_col]
print(f"   Columns in X (after dropping target): {len(X_cols)}")
print(f"   Is target in X columns: {target_col in X_cols}")

if target_col in X_cols:
    print("\n   ❌ LEAKAGE DETECTED!")
    print(f"   The target '{target_col}' appears in the feature columns!")
else:
    print("\n   ✅ No leakage in train/test split")

# Check for duplicate target columns
target_like_cols = [col for col in train.columns if 'readmit' in col.lower()]
print(f"\n4. Columns containing 'readmit': {target_like_cols}")

print("\n" + "="*60)