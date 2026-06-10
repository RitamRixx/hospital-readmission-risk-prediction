# 🏥 Hospital Readmission Risk Prediction — MLOps Pipeline

> An end-to-end production-grade MLOps system that predicts whether a patient will be **readmitted to hospital within 30 days**, built with a focus on automation, reproducibility, and monitoring.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [ML Pipeline (DVC)](#ml-pipeline-dvc)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Experiment Tracking](#experiment-tracking)

---

## Project Overview

This project predicts hospital readmission risk for diabetic patients using clinical data — including lab results, medications, diagnoses, and time in hospital. It is built as a **portfolio MLOps project** demonstrating real-world ML engineering practices.

### What makes this production-grade?

| Concern | Solution |
|---|---|
| Data versioning | DVC + DagsHub |
| Experiment tracking | MLflow on DagsHub |
| Data drift monitoring | Evidently AI |
| Pipeline orchestration | Apache Airflow |
| Model serving | FastAPI on Docker |
| Reproducibility | Docker Compose (single stack) |
| Automated retraining | Airflow weekly DAG |
| Model promotion | Candidate vs. Production comparison |

### Model

- **Algorithm:** LightGBM binary classifier
- **Target:** Readmission within 30 days (`<30` = 1, else 0)
- **Decision threshold:** 0.40 (tuned for recall, not default 0.5)
- **Key features:** Time in hospital, number of lab procedures, inpatient visits, medication changes, insulin status, diagnosis group

---

## Tech Stack

```
Data            PostgreSQL · DVC · DagsHub
ML              LightGBM · scikit-learn · category-encoders · Optuna
Tracking        MLflow · DagsHub
Monitoring      Evidently AI
Serving         FastAPI · Uvicorn · Jinja2
Orchestration   Apache Airflow (LocalExecutor)
Infrastructure  Docker · Docker Compose
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        APACHE AIRFLOW (Weekly)                      │
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐ │
│   │ fetch_new    │───▶│  dvc_repro   │───▶│  pipeline_summary    │ │
│   │ _data        │    │              │    │  (always runs)       │ │
│   └──────────────┘    └──────────────┘    └──────────────────────┘ │
│          │                   │                                      │
│   Pulls 30k records    Runs full ML                                 │
│   from REST API         pipeline                                    │
└─────────────────────────────────────────────────────────────────────┘
          │                   │
          ▼                   ▼
┌─────────────────┐  ┌───────────────────────────────────────────────┐
│  PostgreSQL DB  │  │              DVC PIPELINE STAGES               │
│                 │  │                                                │
│ raw_hospital_   │  │  data_ingestion → data_cleaning →             │
│ data            │  │  feature_engineering → drift_check →          │
│                 │  │  data_split → training_eval →                 │
│ hospital_data_  │  │  model_compare → register_model               │
│ archive         │  │                                                │
└─────────────────┘  └───────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
             ┌──────────┐       ┌────────────┐      ┌────────────┐
             │  MLflow  │       │  Evidently │      │  FastAPI   │
             │ DagsHub  │       │   Drift    │      │  Serving   │
             │ Tracking │       │   Report   │      │ :8000      │
             └──────────┘       └────────────┘      └────────────┘
```

### Data Flow

```
External REST API (Render)
        │
        │  Sliding window — 30k records per weekly run
        ▼
ETL Pipeline (extract → transform → load)
        │
        ▼
PostgreSQL (raw_hospital_data table)
        │
        │  Flattened via SQL VIEW
        ▼
DVC: data_ingestion → CSV → cleaning → feature engineering
        │
        ├──▶ Drift check vs. reference data (Evidently)
        │
        ▼
Train candidate model → compare vs. production model
        │
        ├── If improvement ≥ 0.01 ROC-AUC → deploy candidate
        └── Log to MLflow → register in Model Registry
```

---

## ML Pipeline (DVC)

The pipeline is fully managed by DVC. Each stage caches outputs and only re-runs if its inputs or code changed.

```
data_ingestion
     │  Reads from PostgreSQL view → data/raw/raw.csv
     ▼
data_cleaning
     │  Removes duplicates → data/interim/cleaned_data.csv
     ▼
feature_engineering
     │  ICD-9 diagnosis grouping, age mapping, medication features,
     │  rare category grouping, interaction features
     │  → data/processed/featured_data.csv
     ▼
drift_check
     │  Evidently DataDriftPreset vs. reference_data.csv
     │  → models/drift_decision.json + HTML report
     ▼
data_split
     │  Stratified 80/20 split → data/final/{train,test}.csv
     ▼
training_eval
     │  LightGBM training + MLflow logging
     │  → models/model_candidate.pkl + candidate metrics
     ▼
model_compare
     │  Candidate vs. Production on ROC-AUC + F1
     │  Deploys if improvement ≥ 0.01
     │  → models/deployment_decision.json
     ▼
register_model
     │  Registers winning model to MLflow Model Registry
     └  Promotes to "Production" stage
```

To run the full pipeline manually:

```bash
dvc repro
```

To run a specific stage:

```bash
dvc repro training_eval
```

---

## Project Structure

```
hospital-readmission-risk-prediction/
│
├── app.py                        # FastAPI application
├── params.yaml                   # Single source of truth for all config
├── dvc.yaml                      # DVC pipeline definition
├── dvc.lock                      # DVC pipeline state (committed)
├── docker-compose.yaml           # Full stack: app + Airflow
├── Dockerfile                    # FastAPI app image
├── requirements-api.txt          # API-only dependencies
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py     # Reads from PostgreSQL view
│   │   ├── data_cleaning.py      # Deduplication
│   │   └── etl/
│   │       ├── extract.py        # REST API fetch + sliding window state
│   │       ├── transform.py      # Dedup transform
│   │       ├── load.py           # Load to PostgreSQL (replace/append)
│   │       └── pipeline.py       # ETL orchestrator (full + batch modes)
│   │
│   ├── features/
│   │   ├── feature_engineering.py  # ICD-9 mapping, encodings, new features
│   │   └── data_split.py           # Stratified train/test split
│   │
│   ├── model/
│   │   ├── training_eval.py      # LightGBM train + MLflow logging
│   │   ├── model_compare.py      # Candidate vs. production comparison
│   │   └── register_model.py     # MLflow Model Registry promotion
│   │
│   ├── monitoring/
│   │   └── data_drift.py         # Evidently drift detection
│   │
│   └── db/
│       ├── engine.py             # SQLAlchemy engine factory
│       ├── schema.sql            # Table definitions
│       └── views.sql             # Flattened JSONB view
│
├── dags/
│   └── hospital_pipeline_dag.py  # Airflow DAG (3 tasks)
│
├── models/                       # Serialized models + metrics (DVC tracked)
├── data/                         # All data dirs (DVC tracked, git-ignored)
├── templates/                    # Jinja2 HTML templates
├── static/                       # CSS + JS for the web UI
└── docker-init-scripts/
    └── init-databases.sh         # Creates hospital_readmission DB on first run
```

---

## Setup & Installation

### Prerequisites

- Docker & Docker Compose
- Git
- DVC (`pip install dvc`)

### 1. Clone the repository

```bash
git clone https://github.com/RitamRixx/hospital-readmission-risk-prediction.git
cd hospital-readmission-risk-prediction
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
# PostgreSQL — Hospital DB
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=db
POSTGRES_PORT=5432
POSTGRES_DB=hospital_readmission

# External data API
API=https://your-data-api-url.com/endpoint

# MLflow / DagsHub
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

### 3. Pull model artifacts from DVC

```bash
dvc pull
```

This downloads the trained model files (`models/model.pkl`, encoders, etc.) from DagsHub remote storage.

### 4. Start the full stack

```bash
docker compose up --build
```

This starts:

| Service | URL | Description |
|---|---|---|
| FastAPI app | http://localhost:8000 | Prediction API + web UI |
| Adminer | http://localhost:8080 | Database admin UI |
| Airflow UI | http://localhost:8081 | Pipeline orchestration |

> **Note:** On first run, `airflow-init` creates the Airflow database and admin user (username: `admin`, password: `admin`). This container exits with code 0 after completion — that is expected behaviour.

### 5. Initialize the database schema

On first run only, connect to the hospital database and execute:

```bash
docker exec -it hospital_db psql -U your_user -d hospital_readmission -f /dev/stdin < src/db/schema.sql
docker exec -it hospital_db psql -U your_user -d hospital_readmission -f /dev/stdin < src/db/views.sql
```

### 6. Run the ETL pipeline to load initial data

```bash
# Inside the project root (with dependencies installed), or via docker exec
python -m src.data.etl.pipeline --full 10000 1000 --replace
```

This fetches 10,000 records from the external API in batches of 1,000 and loads them into PostgreSQL.

### 7. Run the ML pipeline

```bash
dvc repro
```

---

## Running the Application

### Web UI

Navigate to **http://localhost:8000** — fill in the patient details form and get an instant readmission risk prediction.

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{"status": "healthy"}
```

---

## API Reference

### `POST /predict`

JSON prediction endpoint.

**Request body:**

```json
{
  "gender": "Male",
  "age": "[70-80)",
  "admission_type_id": "1",
  "discharge_disposition_id": "1",
  "admission_source_id": "7",
  "time_in_hospital": 5,
  "num_lab_procedures": 50,
  "num_medications": 15,
  "number_emergency": 1,
  "number_inpatient": 2,
  "number_diagnoses": 7,
  "metformin": "Steady",
  "change": "Ch",
  "diabetesmed": "Yes",
  "diag_1_group": "Circulatory",
  "num_med_changes": 3,
  "total_visits": 3,
  "insulin_coded": 2,
  "num_med_active": 10
}
```

**Response:**

```json
{
  "readmission_risk": "High Risk",
  "probability": 0.6821,
  "risk_percentage": "68.21%",
  "threshold_used": 0.4,
  "risk_level": "high"
}
```

Risk levels: `low` (< 0.3) · `moderate` (0.3–threshold) · `high` (threshold–0.7) · `very high` (≥ 0.7)

### Interactive API Docs

Available at **http://localhost:8000/docs** (Swagger UI) once the app is running.

---

## Experiment Tracking

All training runs are logged to MLflow on DagsHub:

- **Tracking URI:** `https://dagshub.com/RitamRixx/hospital-readmission-risk-prediction.mlflow`
- **Experiment:** `production-pipeline-New`
- **Logged per run:** accuracy, precision, recall, F1, ROC-AUC, all hyperparameters, model artifact, encoders

The best-performing model is automatically promoted to the `Production` stage in the MLflow Model Registry after passing the candidate vs. production comparison gate.

---

## Automated Retraining (Airflow)

The Airflow DAG `hospital_readmission_pipeline` runs **weekly** and executes three tasks:

```
fetch_new_data  ──▶  dvc_repro  ──▶  pipeline_summary
```

- `fetch_new_data` — pulls the next 30k records from the external API using a sliding window offset stored in `data/etl_state/state.json`
- `dvc_repro` — runs the full DVC ML pipeline; DVC skips unchanged stages automatically
- `pipeline_summary` — prints a run summary (offset, drift decision, deployment decision) regardless of upstream success/failure

Airflow UI: **http://localhost:8081** (admin / admin)

---

## Deployment
 
The FastAPI prediction service is deployed on **GCP Cloud Run** and is publicly accessible.
 
| | |
|---|---|
| **Live URL** | https://prediction-api-1055616729020.us-central1.run.app |
| **Region** | us-central1 |
| **Scaling** | Auto (Min: 0, Max: 10 instances) |
| **Container Registry** | GCP Artifact Registry |
 
### How it was deployed
 
**1. Build and push the Docker image to Artifact Registry**
 
```bash
# Configure Docker to authenticate with GCP
gcloud auth configure-docker us-central1-docker.pkg.dev
 
# Build and tag the image
docker build -t us-central1-docker.pkg.dev/<PROJECT_ID>/hospital-repo/prediction-api:latest .
 
# Push to Artifact Registry
docker push us-central1-docker.pkg.dev/<PROJECT_ID>/hospital-repo/prediction-api:latest
```
 
**2. Deploy to Cloud Run**
 
```bash
gcloud run deploy prediction-api \
  --image us-central1-docker.pkg.dev/<PROJECT_ID>/hospital-repo/prediction-api:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8000
```
 
> Model artifacts (`model.pkl`, encoders, `params.yaml`) are baked into the container image at build time via the `Dockerfile` — no external volume mounts needed on Cloud Run.
 
---

## Author

**Ritam Rakshit** · [GitHub](https://github.com/RitamRixx) · [DagsHub](https://dagshub.com/RitamRixx)

---

*Built as a portfolio MLOps project demonstrating end-to-end ML engineering — from data ingestion to model serving and automated retraining.*