import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split


def load_params():
    try:
        with open("params.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError("Error loading params.yaml") from e


def data_split(input_path, train_path, test_path):
    params = load_params()

    split_cfg = params["data_split"]
    target_cfg = params["feature_engineering"]["target"]

    df = pd.read_csv(input_path)

    if target_cfg["source_column"] not in df.columns:
        raise RuntimeError(
            f"Raw target column '{target_cfg['source_column']}' not found in dataset"
        )

    df[target_cfg["column_name"]] = df[target_cfg["source_column"]].map(
        target_cfg["mapping"]
    )

    if df[target_cfg["column_name"]].nunique() < 2:
        raise RuntimeError("Target has only one class after mapping")

    df = df.drop(columns=[target_cfg["source_column"]])

    X = df.drop(columns=[target_cfg["column_name"]])
    y = df[target_cfg["column_name"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")
    print("Train target distribution:")
    print(y_train.value_counts())
    print("Test target distribution:")
    print(y_test.value_counts())


def main():
    params = load_params()
    data_split(
        params["paths"]["featured_data"],
        params["paths"]["train_data"],
        params["paths"]["test_data"]
    )


if __name__ == "__main__":
    main()
