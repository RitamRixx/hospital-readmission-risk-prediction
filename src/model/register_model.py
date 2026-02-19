import yaml
import mlflow
import dagshub
from mlflow.tracking import MlflowClient

def load_params():
    try:
        with open('params.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise Exception(f'Error loading file params.yaml') from e
    
def register_latest_model():
    params = load_params()
    mlflow_params = params['mlflow']

    dagshub.init(
        repo_owner=mlflow_params['repo_owner'],
        repo_name=mlflow_params['repo_name'],
        mlflow=True
    )

    # Set tracking URI
    mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
    
    client = MlflowClient()
    model_name = mlflow_params["model_name"]


    experiment = mlflow.get_experiment_by_name(mlflow_params['experiment_name'])
    
    print(f"Searching for latest run in experiment: {mlflow_params['experiment_name']}")


    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["start_time DESC"],  
        max_results=1
    ) 

    latest_run = runs.iloc[0]
    run_id = latest_run['run_id']

    print(f"\nLatest Run ID: {run_id}")
    print(f"Run Start Time: {latest_run['start_time']}")
    
    if 'metrics.roc_auc' in latest_run:
        print(f"ROC-AUC: {latest_run['metrics.roc_auc']:.4f}")
    if 'metrics.accuracy' in latest_run:
        print(f"Accuracy: {latest_run['metrics.accuracy']:.4f}")

    try:
        registered_versions = client.search_model_versions(f"name='{model_name}'")
        
        existing_version = None
        for version in registered_versions:
            if version.run_id == run_id:
                existing_version = version
                break

        if existing_version:
            print(f"\nModel from run {run_id} already registered as version {existing_version.version}")
            print(f"Promoting version {existing_version.version} to Production...")
            
            client.transition_model_version_stage(
                name=model_name,
                version=existing_version.version,
                stage="Production",
                archive_existing_versions=True
            )
            
            print(f"Model {model_name} {existing_version.version} promoted to Production")
            
        else:
            print(f"\nRegistering new model version from latest run {run_id}...")
            model_uri = f"runs:/{run_id}/model"
            
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            print(f"Registered model version: {registered_model.version}")
            
            client.transition_model_version_stage(
                name=model_name,
                version=registered_model.version,
                stage="Production",
                archive_existing_versions=True
            )
            
            print(f"Model {model_name} v{registered_model.version} promoted to Production")
            
    except Exception as e:
        print(f"\nNo existing registered model found. Creating new registration...")
        
        model_uri = f"runs:/{run_id}/model"
        
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        print(f"Registered model version: {registered_model.version}")
        
        client.transition_model_version_stage(
            name=model_name,
            version=registered_model.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"Model {model_name} v{registered_model.version} promoted to Production")




def main():
    try:
        register_latest_model()
    except Exception as e:
        print(f"Error during model registration: {e}")
        raise

if __name__ == "__main__":
    main()