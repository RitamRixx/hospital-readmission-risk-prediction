# import pandas as pd
# import yaml
# import os
# from sklearn.model_selection import train_test_split

# def load_params():
#     try:
#         with open('params.yaml', 'r') as f:
#             return yaml.safe_load(f)
#     except Exception as e:
#         raise Exception(f'Error loading file params.yaml') from e
    

# def data_split(input_path, train_path, test_path):
#     params = load_params()

#     data_split = params["data_split"]
#     target_col = params["feature_engineering"]["target"]["column_name"] 

#     df = pd.read_csv(input_path)


#     X = df.drop(target_col, axis=1)
#     y = df[target_col]

#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=data_split['test_size'],random_state=data_split['random_state'],stratify=y)


#     print(f"Training set: {X_train.shape}")
#     print(f"Test set: {X_test.shape}")

#     train_df = pd.concat([X_train, y_train], axis=1)
#     test_df = pd.concat([X_test, y_test], axis=1)
    
#     os.makedirs(os.path.dirname(train_path), exist_ok=True)
#     train_df.to_csv(train_path, index=False)
#     test_df.to_csv(test_path, index=False)
    
#     print(f"Train data saved to: {train_path}")
#     print(f"Test data saved to: {test_path}")


# def main():
#     params = load_params()
#     data_split(
#         params["paths"]["featured_data"],
#         params["paths"]["train_data"],
#         params["paths"]["test_data"]
#     )

# if __name__ == '__main__':
#     main()
#####################################################################################

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
#################################################################################
# import pandas as pd
# import yaml
# import os
# from sklearn.model_selection import train_test_split


# def load_params():
#     try:
#         with open("params.yaml", "r") as f:
#             return yaml.safe_load(f)
#     except Exception as e:
#         raise RuntimeError("Error loading params.yaml") from e


# def data_split(input_path, train_path, test_path):
#     params = load_params()

#     split_cfg = params["data_split"]
#     target_cfg = params["feature_engineering"]["target"]
#     target_col = target_cfg["column_name"]

#     df = pd.read_csv(input_path)

#     print(f"Input shape: {df.shape}")
#     print(f"Columns: {df.columns.tolist()}")

#     # ✅ Target should already exist from feature_engineering.py
#     if target_col not in df.columns:
#         raise RuntimeError(
#             f"Target column '{target_col}' not found! "
#             "It should have been created in feature_engineering.py"
#         )

#     # ✅ Verify target has 2 classes
#     if df[target_col].nunique() < 2:
#         raise RuntimeError(f"Target '{target_col}' has only one class!")

#     print(f"✓ Target column '{target_col}' found")
#     print(f"  Distribution: {df[target_col].value_counts().to_dict()}")

#     # ✅ Split data (target already exists, just separate X and y)
#     X = df.drop(columns=[target_col])
#     y = df[target_col]

#     # ✅ Verify NO target in X
#     if target_col in X.columns:
#         raise RuntimeError(f"LEAKAGE: Target '{target_col}' found in X!")

#     print(f"\n✓ Data prepared for split:")
#     print(f"  X shape: {X.shape}")
#     print(f"  y shape: {y.shape}")
#     print(f"  Target in X: {target_col in X.columns}")  # Should be False

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=split_cfg["test_size"],
#         random_state=split_cfg["random_state"],
#         stratify=y
#     )

#     train_df = pd.concat([X_train, y_train], axis=1)
#     test_df = pd.concat([X_test, y_test], axis=1)

#     os.makedirs(os.path.dirname(train_path), exist_ok=True)
#     train_df.to_csv(train_path, index=False)
#     test_df.to_csv(test_path, index=False)

#     print(f"\n✓ Train data saved to: {train_path}")
#     print(f"  Shape: {train_df.shape}")
#     print(f"  Target distribution:\n{y_train.value_counts()}")

#     print(f"\n✓ Test data saved to: {test_path}")
#     print(f"  Shape: {test_df.shape}")
#     print(f"  Target distribution:\n{y_test.value_counts()}")


# def main():
#     params = load_params()
#     data_split(
#         params["paths"]["featured_data"],
#         params["paths"]["train_data"],
#         params["paths"]["test_data"]
#     )


# if __name__ == "__main__":
#     main()