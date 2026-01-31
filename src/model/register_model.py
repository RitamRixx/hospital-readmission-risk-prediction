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
################################################################################################################

# import yaml
# import mlflow
# import dagshub
# from mlflow.tracking import MlflowClient

# def load_params():
#     try:
#         with open('params.yaml', 'r') as f:
#             return yaml.safe_load(f)
#     except Exception as e:
#         raise Exception(f'Error loading file params.yaml') from e
    
# def register_latest_model():
#     params = load_params()
#     mlflow_params = params['mlflow']

#     dagshub.init(
#         repo_owner=mlflow_params['repo_owner'],
#         repo_name=mlflow_params['repo_name'],
#         mlflow=True
#     )

#     mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
    
#     client = MlflowClient()
#     model_name = mlflow_params["model_name"]

#     experiment = mlflow.get_experiment_by_name(mlflow_params['experiment_name'])
    
#     print("="*60)
#     print("MODEL REGISTRATION")
#     print("="*60)
#     print(f"Experiment: {mlflow_params['experiment_name']}")
#     print(f"Model Name: {model_name}")
#     print("="*60 + "\n")

#     # Get latest run
#     runs = mlflow.search_runs(
#         experiment_ids=[experiment.experiment_id],
#         filter_string="",
#         order_by=["start_time DESC"],  
#         max_results=1
#     ) 

#     if runs.empty:
#         raise Exception("No runs found in experiment!")

#     latest_run = runs.iloc[0]
#     run_id = latest_run['run_id']

#     print(f"Latest Run ID: {run_id}")
#     print(f"Start Time: {latest_run['start_time']}")
    
#     if 'metrics.roc_auc' in latest_run:
#         print(f"ROC-AUC: {latest_run['metrics.roc_auc']:.4f}")
#     if 'metrics.accuracy' in latest_run:
#         print(f"Accuracy: {latest_run['metrics.accuracy']:.4f}")

#     # ==================================================================
#     # CRITICAL: Verify model artifact exists before attempting registration
#     # ==================================================================
#     print("\n" + "="*60)
#     print("VERIFYING MODEL ARTIFACT EXISTS")
#     print("="*60)
    
#     artifacts = client.list_artifacts(run_id)
    
#     model_artifact_found = False
#     print("Artifacts in run:")
#     for artifact in artifacts:
#         icon = "📁" if artifact.is_dir else "📄"
#         print(f"  {icon} {artifact.path}")
        
#         if artifact.path == "model" and artifact.is_dir:
#             model_artifact_found = True
#             # List contents
#             try:
#                 model_contents = client.list_artifacts(run_id, "model")
#                 print("    Model directory contents:")
#                 for sub in model_contents:
#                     print(f"      📄 {sub.path}")
#             except Exception as e:
#                 print(f"      ⚠️ Could not list contents: {e}")
    
#     if not model_artifact_found:
#         print("\n" + "="*60)
#         print("❌ ERROR: MODEL ARTIFACT NOT FOUND")
#         print("="*60)
#         print("The model directory does not exist in this run.")
#         print("Registration cannot proceed.")
#         print("\nPossible causes:")
#         print("1. Model logging failed during training")
#         print("2. DagsHub sync is delayed (try again in a few minutes)")
#         print("3. Network issues prevented upload")
#         print("\nSolution:")
#         print("- Re-run the training script")
#         print("- Check MLflow UI to confirm model is visible")
#         print("="*60)
#         raise Exception(f"Model artifact not found in run {run_id}")
    
#     print("\n✓ Model artifact verified!")
#     print("="*60 + "\n")

#     # ==================================================================
#     # REGISTRATION
#     # ==================================================================
#     print("="*60)
#     print("REGISTERING MODEL")
#     print("="*60)
    
#     model_uri = f"runs:/{run_id}/model"
#     print(f"Model URI: {model_uri}\n")

#     try:
#         # Check if this run is already registered
#         registered_versions = client.search_model_versions(f"name='{model_name}'")
        
#         existing_version = None
#         for version in registered_versions:
#             if version.run_id == run_id:
#                 existing_version = version
#                 break

#         if existing_version:
#             print(f"ℹ️  Model from run {run_id} already registered")
#             print(f"   Version: {existing_version.version}")
#             print(f"   Current stage: {existing_version.current_stage}")
#             print(f"\nPromoting version {existing_version.version} to Production...")
            
#             client.transition_model_version_stage(
#                 name=model_name,
#                 version=existing_version.version,
#                 stage="Production",
#                 archive_existing_versions=True
#             )
            
#             print(f"✓ Version {existing_version.version} promoted to Production")
            
#         else:
#             # Register new version
#             print("Registering new model version...")
            
#             try:
#                 registered_model = mlflow.register_model(
#                     model_uri=model_uri,
#                     name=model_name
#                 )
                
#                 print(f"✓ Registered as version {registered_model.version}")
                
#                 # Promote to production
#                 print(f"\nPromoting version {registered_model.version} to Production...")
#                 client.transition_model_version_stage(
#                     name=model_name,
#                     version=registered_model.version,
#                     stage="Production",
#                     archive_existing_versions=True
#                 )
                
#                 print(f"✓ Version {registered_model.version} promoted to Production")
                
#             except Exception as e:
#                 print(f"\n❌ Registration failed: {e}")
                
#                 # If model doesn't exist, create it
#                 if "RESOURCE_DOES_NOT_EXIST" in str(e):
#                     print("\nAttempting to create new registered model...")
                    
#                     registered_model = mlflow.register_model(
#                         model_uri=model_uri,
#                         name=model_name
#                     )
                    
#                     print(f"✓ Created and registered as version {registered_model.version}")
                    
#                     client.transition_model_version_stage(
#                         name=model_name,
#                         version=registered_model.version,
#                         stage="Production",
#                         archive_existing_versions=True
#                     )
                    
#                     print(f"✓ Version {registered_model.version} promoted to Production")
#                 else:
#                     raise
                    
#     except Exception as e:
#         print(f"\n❌ Error during registration: {e}")
#         raise
    
#     print("\n" + "="*60)
#     print("✓ REGISTRATION COMPLETE")
#     print("="*60)
#     print(f"Model: {model_name}")
#     print(f"Run ID: {run_id}")
#     print(f"Stage: Production")
#     print("="*60)


# def main():
#     try:
#         register_latest_model()
#     except Exception as e:
#         print(f"\n{'='*60}")
#         print("REGISTRATION FAILED")
#         print("="*60)
#         print(f"Error: {e}")
#         print("="*60)
#         raise

# if __name__ == "__main__":
#     main()