import numpy as np
import pandas as pd
import mlflow
import dagshub
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,precision_recall_curve
from lightgbm import LGBMClassifier
import category_encoders as ce

dagshub.init(repo_owner='RitamRixx', repo_name='hospital-readmission-risk-prediction', mlflow=True)
mlflow.set_experiment("experiment3")
mlflow.set_tracking_uri("https://dagshub.com/RitamRixx/hospital-readmission-risk-prediction.mlflow")


df = pd.read_csv("notebooks/dataset/data_final10ktest.csv")


num_cols = ["age", "time_in_hospital", "num_lab_procedures","num_medications","number_emergency","number_inpatient", "number_diagnoses","total_visits", "num_med_changes","insulin_coded", "interaction_visits_meds"]
binary_cols = ["gender", "diabetesMed", "metformin", "change"]
diag_grp_col = ["diag_1_group"]
target_encode_cols = ["admission_type_id","discharge_disposition_id","admission_source_id"]

X = df.drop("readmitted_binary", axis=1)
y = df["readmitted_binary"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = neg / pos


label_encoders = {}
for col in binary_cols + diag_grp_col:
    le = LabelEncoder()
    le.fit(X_train[col].astype(str))
    label_encoders[col] = le

def apply_label_encoding(X, encoders):
    X_enc = X.copy()
    for col, le in encoders.items():
        X_enc[col] = le.transform(X_enc[col].astype(str))
    return X_enc

X_train_le = apply_label_encoding(X_train, label_encoders)
X_test_le = apply_label_encoding(X_test, label_encoders)

target_encoder = ce.TargetEncoder(
    cols=target_encode_cols,
    smoothing=10,
    min_samples_leaf=50
)
target_encoder.fit(X_train_le[target_encode_cols], y_train)

X_train_final = X_train_le.copy()
X_test_final = X_test_le.copy()
X_train_final[target_encode_cols] = target_encoder.transform(X_train_le[target_encode_cols])
X_test_final[target_encode_cols] = target_encoder.transform(X_test_le[target_encode_cols])

model = LGBMClassifier(
    objective="binary",
    is_unbalance=True,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1  
)


with mlflow.start_run(run_name="LightGBM_ThresholdTuned"):
    
    model.fit(X_train_final, y_train)
    y_prob = model.predict_proba(X_test_final)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_optimal = (y_prob >= best_threshold).astype(int)
    
    print("Results with DEFAULT threshold (0.5):")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_default):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_default, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_default):.4f}")
    print(f"  F1:        {f1_score(y_test, y_pred_default):.4f}")
    
    print(f"\nResults with OPTIMAL threshold ({best_threshold:.3f}):")
    acc = accuracy_score(y_test, y_pred_optimal)
    prec = precision_score(y_test, y_pred_optimal, zero_division=0)
    rec = recall_score(y_test, y_pred_optimal)
    f1 = f1_score(y_test, y_pred_optimal)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}  ← Catching {rec*100:.1f}% of readmissions!")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    mlflow.log_metric("optimal_threshold", best_threshold)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_param("model", "LightGBM")
    mlflow.sklearn.log_model(model, "lightgbm_model")
    mlflow.log_artifact(__file__)
    mlflow.set_tag("experiment", "lgbm_thrtune")
    mlflow.set_tag("author", "Ritam")    

