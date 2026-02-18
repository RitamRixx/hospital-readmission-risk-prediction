import pandas as pd
import yaml
import json
import os
import pickle
import mlflow
import dagshub
import lightgbm as lgb
import tempfile
import shutil

from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_params():
    try:
        with open('params.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise Exception(f'Error loading file params.yaml') from e

def prepare_data(train_df, test_df, fe_cfg, encode_cfg):
    params = load_params()

    target_col = fe_cfg["target"]["column_name"]
    label_cols = encode_cfg["label_encode_cols"]
    te_cols = encode_cfg["target_encode_cols"]

    X_train = train_df.drop(columns=[target_col]).copy()
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col]).copy()
    y_test = test_df[target_col]

    label_encoders = {}

    for col in label_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = X_test[col].astype(str).apply(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le

    te = TargetEncoder(cols=te_cols, smoothing=10, min_samples_leaf=50)
    X_train[te_cols] = te.fit_transform(X_train[te_cols], y_train)
    X_test[te_cols] = te.transform(X_test[te_cols])

    return X_train, X_test, y_train, y_test, label_encoders, te


def train_model():
    params = load_params()

    print(f"LightGBM version: {lgb.__version__}")
    print(f"MLflow version: {mlflow.__version__}")

    dagshub.init(
        repo_owner=params["mlflow"]["repo_owner"],
        repo_name=params["mlflow"]["repo_name"],
        mlflow=True
    )

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    train_df = pd.read_csv(params["paths"]["train_data"])
    test_df = pd.read_csv(params["paths"]["test_data"])

    threshold = params["model_training"]["optimal_threshold"]
    fe_cfg = params["feature_engineering"]
    encode_cfg = params["encoding"]

    X_train, X_test, y_train, y_test, label_encoders, target_encoder = prepare_data(
        train_df, test_df, fe_cfg, encode_cfg
    )

    with mlflow.start_run(run_name="lightgbm-model") as run:

        model_params = params["model_training"]

        # Train model
        model = lgb.LGBMClassifier(**model_params["lightgbm_params"])
        model.fit(X_train, y_train)

        # Predictions
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        eval_metrics = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }

        # Log parameters
        mlflow.log_params(model_params["lightgbm_params"])
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("optimal_threshold", threshold)

        # Set tags
        mlflow.set_tag("stage", "production")
        mlflow.set_tag("author", "Ritam")
        mlflow.set_tag("pipeline_stage", "training")

        # Save artifacts locally
        os.makedirs(os.path.dirname(params["paths"]["model_output"]), exist_ok=True)
        
        with open(params["paths"]["model_output"], "wb") as f:
            pickle.dump(model, f)

        with open(params["paths"]["label_encoders"], "wb") as f:
            pickle.dump(label_encoders, f)

        with open(params["paths"]["target_encoder"], "wb") as f:
            pickle.dump(target_encoder, f)

        os.makedirs(os.path.dirname(params["paths"]["metrics_output"]), exist_ok=True)
        with open(params["paths"]["metrics_output"], "w") as f:
            json.dump(eval_metrics, f, indent=4)
        
        print("LOGGING MODEL")
        

        
        # Create temp directory for model
        temp_model_dir = tempfile.mkdtemp()
        model_save_path = os.path.join(temp_model_dir, "model")
        
        try:
            print("Saving model with MLflow format...")
            
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # This creates the model files locally
                mlflow.sklearn.save_model(
                    sk_model=model,
                    path=model_save_path
                )
            
            print(f"✓ Model saved to: {model_save_path}")
            
            # List what was created
            print("Model directory contents:")
            for item in os.listdir(model_save_path):
                print(f"  - {item}")

            print("\nUploading model directory...")
            
            for root, dirs, files in os.walk(model_save_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Get relative path within model directory
                    rel_path = os.path.relpath(file_path, model_save_path)
                    artifact_path = os.path.join("model", rel_path).replace("\\", "/")
                    
                    mlflow.log_artifact(file_path, artifact_path=os.path.dirname(artifact_path))
                    print(f"  ✓ Uploaded: {artifact_path}")
            
            print("✓ Model logged successfully")
            
        finally:
            # Cleanup
            if os.path.exists(temp_model_dir):
                shutil.rmtree(temp_model_dir)
        
        print("="*60 + "\n")

        # Log other artifacts
        mlflow.log_artifact(__file__)
        mlflow.log_artifact(params["paths"]["label_encoders"], artifact_path="encoders")
        mlflow.log_artifact(params["paths"]["target_encoder"], artifact_path="encoders")
        print("✓ Other artifacts logged\n")

        # Final verification
        print("="*60)
        print("FINAL VERIFICATION")
        print("="*60)
        
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        run_id = run.info.run_id
        
        import time
        print("Waiting for sync...")
        time.sleep(3)
        
        artifacts = client.list_artifacts(run_id)
        
        print("Artifacts in run:")
        for artifact in artifacts:
            icon = "📁" if artifact.is_dir else "📄"
            print(f"  {icon} {artifact.path}")
            
            if artifact.is_dir:
                try:
                    sub_artifacts = client.list_artifacts(run_id, artifact.path)
                    for sub in sub_artifacts:
                        print(f"    📄 {sub.path}")
                except:
                    pass
        
        print("="*60 + "\n")

        # Results
        print("="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"\nRun ID: {run_id}")
        print("="*60)


def main():
    train_model()


if __name__ == "__main__":
    main()