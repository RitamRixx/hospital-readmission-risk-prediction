import pandas as pd
import yaml
import os
import json
import shutil
from datetime import datetime
import pickle
from sklearn.metrics import f1_score, roc_auc_score


def load_params():
    try:
        with open('params.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise Exception(f'Error loading file params.yaml') from e
    
def load_model(model_path, label_encoder_path, target_encoder_path):
    try:
        with open(model_path,'rb') as f:
            model = pickle.load(f)

        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        with open(target_encoder_path, 'rb') as f:
            target_encoder =pickle.load(f)


        return model, label_encoder, target_encoder
    except Exception as e:
        raise Exception("Error loading model") from e
    

def prepare_test_data(test_df, encoders, te, target_col, label_cols, te_cols):
    X = test_df.drop(columns=[target_col]).copy()
    y = test_df[target_col]

    for col in label_cols:
        X[col] = X[col].astype(str).apply(
            lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0]
        )
        X[col] = encoders[col].transform(X[col])

    X[te_cols] = te.transform(X[te_cols])

    return X, y

def evaluate_model(model, X, y, threshold):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    return {
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, prob)
    }


def safe_replace(src, dest):
    temp_dest = dest + ".tmp"
    shutil.copy(src, temp_dest)
    os.replace(temp_dest, dest)

def update_reference_data(params):
    featured_data_path = params["paths"]["featured_data"]
    reference_path = params["paths"]["reference_data"]

    os.makedirs(os.path.dirname(reference_path), exist_ok=True)
    shutil.copy(featured_data_path, reference_path)

    print(f"Reference data updated: {reference_path}")


def compare_models():

    params = load_params()

    candidate_path = params["paths"]["model_candidate"]
    candidate_enc = params["paths"]["candidate_label_encoders"]
    candidate_te = params["paths"]["candidate_target_encoder"]

    prod_path = params["paths"]["model_output"]
    prod_enc = params["paths"]["label_encoders"]
    prod_te = params["paths"]["target_encoder"]

    test_path = params["paths"]["test_data"]

    primary_metric = params["model_comparison"]["primary_metric"]
    min_improvement = params["model_comparison"]["min_improvement"]
    threshold = params["model_comparison"]["threshold"]

    target_col = params["feature_engineering"]["target"]["column_name"]
    label_cols = params["encoding"]["label_encode_cols"]
    te_cols = params["encoding"]["target_encode_cols"]

    os.makedirs("models", exist_ok=True)


    if not os.path.exists(prod_path):
        print("First deployment — deploying candidate automatically.")

        safe_replace(candidate_path, prod_path)
        safe_replace(candidate_enc, prod_enc)
        safe_replace(candidate_te, prod_te)
        update_reference_data(params)

        decision = {
            "deploy": True,
            "reason": "First deployment",
            "timestamp": str(datetime.utcnow())
        }

        with open("models/deployment_decision.json", "w") as f:
            json.dump(decision, f, indent=4)

        return decision

    test_df = pd.read_csv(test_path)

    cand_model, cand_enc, cand_te = load_model(
        candidate_path, candidate_enc, candidate_te
    )

    X_cand, y = prepare_test_data(
        test_df, cand_enc, cand_te, target_col, label_cols, te_cols
    )

    cand_metrics = evaluate_model(cand_model, X_cand, y, threshold)

    prod_model, prod_enc_data, prod_te_data = load_model(
        prod_path, prod_enc, prod_te
    )

    X_prod, _ = prepare_test_data(
        test_df, prod_enc_data, prod_te_data, target_col, label_cols, te_cols
    )

    prod_metrics = evaluate_model(prod_model, X_prod, y, threshold)

    improvement = cand_metrics[primary_metric] - prod_metrics[primary_metric]

    print("\nMODEL COMPARISON")
    print(f"Primary Metric: {primary_metric}")
    print(f"Production {primary_metric}: {prod_metrics[primary_metric]:.4f}")
    print(f"Candidate  {primary_metric}: {cand_metrics[primary_metric]:.4f}")
    print(f"Improvement: {improvement:+.4f}")
    print(f"Production AUC: {prod_metrics['roc_auc']:.4f}")
    print(f"Candidate  AUC: {cand_metrics['roc_auc']:.4f}")


    deploy = False
    reason = ""

    if improvement >= min_improvement:
        if cand_metrics["roc_auc"] >= prod_metrics["roc_auc"] - 0.01:
            deploy = True
            reason = f"{primary_metric} improved by {improvement:.4f}"
        else:
            reason = "Primary improved but AUC degraded"
    else:
        reason = f"Improvement {improvement:.4f} < threshold {min_improvement}"


    if deploy:
        print(f"DEPLOYING — {reason}")

        safe_replace(candidate_path, prod_path)
        safe_replace(candidate_enc, prod_enc)
        safe_replace(candidate_te, prod_te)

    else:
        print(f"REJECTED — {reason}")

    decision = {
        "deploy": deploy,
        "reason": reason,
        "primary_metric": primary_metric,
        "prod_metrics": prod_metrics,
        "cand_metrics": cand_metrics,
        "improvement": improvement,
        "timestamp": str(datetime.utcnow())
    }

    with open("models/deployment_decision.json", "w") as f:
        json.dump(decision, f, indent=4)

    return decision

if __name__ == "__main__":
    try:
        compare_models()
    except Exception as e:
        print(f"Error during model comparison: {e}")
        exit(1)