import pandas as pd
import yaml
import os
import json
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def load_params(config_path: str = "params.yaml") -> dict:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"Error loading file params.yaml") from e
    

def detect_data_drift():
    params = load_params()
    reference_path = params["paths"]["reference_data"]
    current_path = params["paths"]["featured_data"]


    drift_decision_path = "models/drift_decision.json"
    drift_report_dir = "models/drift_report"

    os.makedirs(os.path.dirname(drift_decision_path), exist_ok=True)
    os.makedirs(drift_report_dir, exist_ok=True)

    if not os.path.exists(reference_path):
        print("=" * 60)
        print("DRIFT CHECK - FIRST RUN")
        print("No reference data found")
        print(f"Reference path: {reference_path}")
        print("\nThis is the first run - skipping drift check")
        print("Model will train on current data")
        print("Current data will be saved as reference after training")


        decision = {
            'drift_detected': True,
            'first_run': True,
            'drift_share': None,
            'dataset_drift': None,
            'timestamp': datetime.now().isoformat(),
            'message': 'First run - no reference data available'
        }

        with open(drift_decision_path, 'w') as f:
            json.dump(decision, f, indent=4)
        
        placeholder_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Drift Report - First Run</title></head>
        <body>
            <h1>Data Drift Check - First Run</h1>
            <p><strong>Status:</strong> No reference data available</p>
            <p><strong>Action:</strong> Training will proceed and create reference data</p>
            <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body>
        </html>
        """
        
        report_path = os.path.join(drift_report_dir, "first_run_no_reference.html")
        with open(report_path, 'w') as f:
            f.write(placeholder_html)
        
        return decision
    
    print("=" * 60)
    print("DRIFT CHECK - STARTED")  


    try:
        reference_df = pd.read_csv(reference_path)
        current_df = pd.read_csv(current_path)

        print(f"Reference data shape: {reference_df.shape}")
        print(f"Current data shape: {current_df.shape}")


    except Exception as e:
        print(f"Error loading data: {e}")
        raise

    if reference_df.empty or current_df.empty:
        raise ValueError("Reference or current dataset is empty. Cannot perform drift check.")
    

    ref_cols = set(reference_df.columns)
    curr_cols = set(current_df.columns)

    if ref_cols != curr_cols:
        print("\nColumn mismatch detected")
        print(f"Missing in current: {ref_cols - curr_cols}")
        print(f"Extra in current: {curr_cols - ref_cols}")
        
        common_cols = list(ref_cols & curr_cols)
        reference_df = reference_df[common_cols]
        current_df = current_df[common_cols]
        print(f"Using {len(common_cols)} common columns")


    drift_threshold = params.get("drift_monitoring", {}).get("thresholds", {}).get("drift_share", 0.2)

    print(f"\nRunning Evidently drift detection...")
    print(f"Drift threshold: {drift_threshold * 100}% of features")


    try:
        report = Report(metrics=[
            DataDriftPreset()
        ])


        report.run(reference_data=reference_df, current_data=current_df)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_report_path = os.path.join(drift_report_dir, f"drift_report_{timestamp}.html")
        # report.save(html_report_path)
        # with open(html_report_path, "w", encoding="utf-8") as f:
        #     f.write(report.get_html())
        report.save_html(html_report_path)
        # html_content = report._repr_html_()
        # with open(html_report_path, "w", encoding="utf-8") as f:
        #     f.write(html_content)
        print(f"HTML report saved: {html_report_path}")


        report_dict = report.as_dict()

        drift_metrics = report_dict['metrics'][0]['result']
        dataset_drift = drift_metrics.get('dataset_drift', False)
        drift_share = drift_metrics.get('drift_share', 0.0)
        number_of_drifted_columns = drift_metrics.get('number_of_drifted_columns', 0)
        
        print("\n" + "=" * 70)
        print("DRIFT ANALYSIS RESULTS")
        print("=" * 70)
        print(f"Dataset Drift Detected: {'YES' if dataset_drift else 'NO'}")
        print(f"Drift Share: {drift_share * 100:.1f}%")
        print(f"Drifted Features: {number_of_drifted_columns}")

        if 'drift_by_columns' in drift_metrics and number_of_drifted_columns > 0:
            print("\nDrifted Features:")
            for col, col_drift in drift_metrics['drift_by_columns'].items():
                if col_drift.get('drift_detected', False):
                    drift_score = col_drift.get('drift_score', 'N/A')
                    print(f"{col}: {drift_score}")

        if drift_share >= drift_threshold or dataset_drift:
            print(f"\nDRIFT DETECTED - Retraining will proceed")
            drift_decision = True
        else:
            print(f"\nNO SIGNIFICANT DRIFT - Keeping current model")
            drift_decision = False


        decision = {
            'drift_detected': drift_decision,
            'first_run': False,
            'dataset_drift': dataset_drift,
            'drift_share': drift_share,
            'number_of_drifted_columns': number_of_drifted_columns,
            'drift_threshold': drift_threshold,
            'timestamp': datetime.now().isoformat(),
            'reference_records': len(reference_df),
            'current_records': len(current_df),
            'html_report': html_report_path
        }


        with open(drift_decision_path, 'w') as f:
            json.dump(decision, f, indent=4)
        
        print(f"\nDrift decision saved: {drift_decision_path}")
        print("=" * 70)
        
        return decision
        
    except Exception as e:
        print(f"\nError during drift detection: {e}")
        raise

def main():
    try:
        decision = detect_data_drift()
        
        if decision['drift_detected']:
            print("\nPipeline will continue to training")
            exit(0)  
        else:
            print("\nPipeline will stop (no retraining needed)")
            exit(0)
            
    except Exception as e:
        print(f"\nDrift check failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
