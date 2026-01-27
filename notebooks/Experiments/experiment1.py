import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


dagshub.init(repo_owner='RitamRixx', repo_name='hospital-readmission-risk-prediction', mlflow=True)
mlflow.set_experiment("exp1_rf_BaseModel")
mlflow.set_tracking_uri("https://dagshub.com/RitamRixx/hospital-readmission-risk-prediction.mlflow")



df = pd.read_csv("notebooks/dataset/data_final10ktest.csv")

X = df.drop("readmitted", axis=1)
y = df["readmitted"]

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
         categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ],
    remainder='passthrough'
)

n_estimators = 100

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators, random_state=42,n_jobs=-1))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="Basemode(RF)"):
    model = rf_pipeline.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)


    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc_auc)


    for k, v in model.get_params().items():
        mlflow.log_param(k, v)


    mlflow.set_tag("model", "randomforest")
    mlflow.set_tag("author", "Ritam")
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(model, "Randomforest_model")

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("ROC AUC:", roc_auc)
