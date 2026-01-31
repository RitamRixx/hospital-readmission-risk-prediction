# import json
# import optuna
# import mlflow
# import mlflow.lightgbm
# import dagshub
# import pandas as pd
# import numpy as np

# from lightgbm import LGBMClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import recall_score,precision_score,roc_auc_score,confusion_matrix
# from category_encoders import TargetEncoder

# dagshub.init(repo_owner='RitamRixx', repo_name='hospital-readmission-risk-prediction', mlflow=True)
# mlflow.set_experiment("experiment4")
# mlflow.set_tracking_uri("https://dagshub.com/RitamRixx/hospital-readmission-risk-prediction.mlflow")



# path = "notebooks/dataset/data_final10ktest.csv"
# exp_name = "exp4_lgbm_optuna"
# random_state = 42
# n_trials = 50

# target_col = "readmitted_binary"

# label_encode = [
#     "gender",
#     "diabetesMed",
#     "metformin",
#     "change",
#     "diag_1_group"
# ]

# target_encode = [
#     "admission_type_id",
#     "discharge_disposition_id",
#     "admission_source_id"
# ]

# df = pd.read_csv(path)

# X = df.drop(columns=[target_col])
# y = df[target_col]

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     stratify=y,
#     random_state=random_state
# )


# def encode_data(X_train, X_test, y_train):
#     X_train = X_train.copy()
#     X_test = X_test.copy()

#     label_encoders = {}

#     for col in label_encode:
#         le = LabelEncoder()
#         X_train[col] = le.fit_transform(X_train[col].astype(str))
#         X_test[col] = le.transform(X_test[col].astype(str))
#         label_encoders[col] = le

#     te = TargetEncoder(
#         cols=target_encode,
#         smoothing=10,
#         min_samples_leaf=50
#     )
#     X_train[target_encode] = te.fit_transform(
#         X_train[target_encode], y_train
#     )
#     X_test[target_encode] = te.transform(
#         X_test[target_encode]
#     )

#     return X_train, X_test


# X_train, X_test = encode_data(X_train, X_test, y_train)


# def objective(trial):

#     params = {
#         "objective": "binary",
#         "is_unbalance": True,
#         "n_estimators": trial.suggest_int("n_estimators", 200, 800),
#         "num_leaves": trial.suggest_int("num_leaves", 20, 80),
#         "max_depth": trial.suggest_int("max_depth", 3, 15),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
#         "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "random_state": random_state,
#         "n_jobs": -1,
#         "verbose": -1
#     }

#     model = LGBMClassifier(**params)
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     recall = recall_score(y_test, y_pred)

#     mlflow.log_metric("trial_recall", recall)

#     return recall



# with mlflow.start_run(run_name="exp4_optuna_search"):

#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=n_trials)

#     best_params = {
#         "objective": "binary",
#         "is_unbalance": True,
#         "random_state": random_state,
#         **study.best_params
#     }

#     # ---- Train final model
#     final_model = LGBMClassifier(**best_params)
#     final_model.fit(X_train, y_train)

#     threshold = 0.62

#     y_prob = final_model.predict_proba(X_test)[:, 1]
#     y_pred = (y_prob >= threshold).astype(int)


#     recall = recall_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_prob)
#     cm = confusion_matrix(y_test, y_pred)

#     mlflow.log_params(best_params)
#     mlflow.log_metric("final_recall", recall)
#     mlflow.log_metric("final_precision", precision)
#     mlflow.log_metric("final_roc_auc", roc_auc)

#     mlflow.lightgbm.log_model(final_model, "model")

#     with open("best_params_exp4.json", "w") as f:
#         json.dump(best_params, f, indent=4)

#     mlflow.log_artifact("best_params_exp4.json")

#     # ---- Console output
#     print("Best Hyperparameters:")
#     print(json.dumps(best_params, indent=2))
#     print("\nConfusion Matrix:")
#     print(cm)
#     print(f"\nRecall:    {recall:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"ROC-AUC:   {roc_auc:.4f}")


import json
import optuna
import mlflow
import mlflow.lightgbm
import dagshub
import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix, f1_score
from category_encoders import TargetEncoder

dagshub.init(repo_owner='RitamRixx', repo_name='hospital-readmission-risk-prediction', mlflow=True)
mlflow.set_experiment("experiment5")
mlflow.set_tracking_uri("https://dagshub.com/RitamRixx/hospital-readmission-risk-prediction.mlflow")

path = "notebooks/dataset/data_final10ktest.csv"
exp_name = "exp5_lgbm_optuna_fixed"
random_state = 42
n_trials = 50

target_col = "readmitted_binary"

label_encode = [
    "gender",
    "diabetesMed",
    "metformin",
    "change",
    "diag_1_group"
]

target_encode = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id"
]

df = pd.read_csv(path)

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=random_state
)


def encode_data(X_train, X_test, y_train):
    X_train = X_train.copy()
    X_test = X_test.copy()

    label_encoders = {}

    for col in label_encode:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        
        # Handle unseen categories
        X_test[col] = X_test[col].astype(str).apply(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le

    te = TargetEncoder(
        cols=target_encode,
        smoothing=10,
        min_samples_leaf=50
    )
    X_train[target_encode] = te.fit_transform(
        X_train[target_encode], y_train
    )
    X_test[target_encode] = te.transform(
        X_test[target_encode]
    )

    return X_train, X_test


X_train, X_test = encode_data(X_train, X_test, y_train)


def objective(trial):
    
    params = {
        "objective": "binary",
        "class_weight": "balanced",  # Changed from is_unbalance
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "num_leaves": trial.suggest_int("num_leaves", 20, 80),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1
    }

    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log all metrics for this trial
    mlflow.log_metric("trial_recall", recall)
    mlflow.log_metric("trial_precision", precision)
    mlflow.log_metric("trial_f1", f1)

    # Optimize F1 instead of just recall for better balance
    return f1


with mlflow.start_run(run_name="exp5_optuna_search_fixed"):

    # Create study - optimize F1 for balanced results
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = {
        "objective": "binary",
        "class_weight": "balanced",
        "random_state": random_state,
        **study.best_params
    }

    # Train final model with best params
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL WITH BEST PARAMS")
    print("="*60)
    
    final_model = LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # Use 0.5 as default threshold
    threshold = 0.5

    y_prob = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate all metrics
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Log to MLflow
    mlflow.log_params(best_params)
    mlflow.log_metric("final_recall", recall)
    mlflow.log_metric("final_precision", precision)
    mlflow.log_metric("final_f1", f1)
    mlflow.log_metric("final_roc_auc", roc_auc)
    mlflow.log_metric("threshold", threshold)

    mlflow.lightgbm.log_model(final_model, "model")

    # Save best params
    with open("best_params_exp5.json", "w") as f:
        json.dump(best_params, f, indent=4)

    mlflow.log_artifact("best_params_exp5.json")

    # Print results
    print("\nBest Hyperparameters:")
    print(json.dumps(best_params, indent=2))
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nMetrics:")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("="*60)
