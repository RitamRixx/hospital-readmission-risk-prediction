from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import uvicorn
import pickle
import yaml


model = None
label_encoders = None
target_encoder = None
params = None

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, label_encoders, target_encoder, params
    
    try:
        print("Loading params.yaml...")
        params = load_params()
        print(f"Params loaded: {list(params.keys())}")
        
        print("Loading model.pkl...")
        with open("models/model.pkl", 'rb') as f:
            model = pickle.load(f)
        print("Model loaded")
        
        print("Loading label_encoders.pkl...")
        with open('models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        print("Label encoders loaded")
        
        print("Loading target_encoder.pkl...")
        with open('models/target_encoder.pkl', 'rb') as f:
            target_encoder = pickle.load(f)
        print("Target encoder loaded")
        
        print("All models loaded successfully")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error loading models: {e}")
    
    yield

app = FastAPI(title='hospital-readmission-risk-pred-api',
              description="API for predicting hospital readmisson",
              version='1.0.0',
              lifespan=lifespan
              )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class hospitalreadmissionriskInput(BaseModel):
    gender: Literal['Male', 'Female']=Field(..., description="patient's gender")
    age: Literal['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)','[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']=Field(..., description="patient's age group")
    admission_type_id: Annotated[str, Field(..., description="Type of admission")]
    discharge_disposition_id: Annotated[str, Field(..., description="Discharge disposition")]
    admission_source_id: Annotated[str, Field(..., description="Source of admission")]
    time_in_hospital: Annotated[int, Field(..., ge=1, le=14, description="Days in hospital")]
    num_lab_procedures: Annotated[int, Field(..., ge=1, le=200, description="Number of lab procedures")]
    num_medications: Annotated[int, Field(..., ge=0, description="Number of medications")]
    number_emergency: Annotated[int, Field(..., ge=0, description="Emergency visits")]
    number_inpatient: Annotated[int, Field(..., ge=0, description="Inpatient visits")]
    number_diagnoses: Annotated[int, Field(..., ge=0, description="Number of diagnoses")]
    metformin: Literal['No', 'Down', 'Steady', 'Up'] = Field(..., description="Metformin status")
    change: Literal['Ch', 'No'] = Field(..., description="Medication change")
    diabetesmed: Literal['Yes', 'No'] = Field(..., description="On diabetes medication")
    diag_1_group: Literal[
        'Diabetes', 'Circulatory', 'Respiratory', 'Digestive', 
        'Genitourinary', 'Cancer', 'Musculoskeletal', 'Injury', 
        'Other', 'Unknown'
    ] = Field(..., description="Primary diagnosis category")
    num_med_changes: Annotated[int, Field(..., ge=0, description="Number of medication changes")]
    total_visits: Annotated[int, Field(..., ge=0, description="Total hospital visits")]
    insulin_coded: Literal[0, 1, 2, 3] = Field(..., description="Insulin: 0=No, 1=Down, 2=Steady, 3=Up")
    num_med_active: Annotated[int, Field(..., ge=0, description="Number of active medications")]
    

class PredictionResponse(BaseModel):
    readmission_risk: str
    probability: float
    risk_percentage: str
    threshold_used: float
    risk_level: str


def apply_rare_category_grouping(df: pd.DataFrame) -> pd.DataFrame:

    rare_cfg = params['feature_engineering']['rare_category_grouping']
    
    if not rare_cfg['enabled']:
        return df
    
    threshold = rare_cfg['threshold']
    cols_to_group = rare_cfg['columns']
    group_name = rare_cfg['group_name']
    

    valid_categories = {
        'admission_type_id': ['1', '2', '3', '5', '6'],
        'discharge_disposition_id': ['1', '2', '3', '5', '6', '11', '18', '22'],
        'admission_source_id': ['1', '2', '4', '6', '7', '17']
    }
    for col in cols_to_group:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
            df[col] = df[col].apply(
                lambda x: x if x in valid_categories.get(col, []) else group_name
            )
    
    return df


def engineer_features(input_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_dict])

    df = apply_rare_category_grouping(df)

    age_mapping = params['feature_engineering']['age_mapping']

    df['age'] = df['age'].map(age_mapping)

    df['interaction_visits_meds'] = df['total_visits'] * df['num_med_active']

    df = df.drop(columns=['num_med_active'])

    model_features = [
        'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
        'num_medications', 'number_emergency', 'number_inpatient',
        'number_diagnoses', 'metformin', 'change', 'diabetesmed',
        'diag_1_group', 'num_med_changes', 'total_visits',
        'interaction_visits_meds', 'insulin_coded'
    ]

    df = df[model_features]

    return df

def apply_encodings(df: pd.DataFrame) -> pd.DataFrame:
    
    label_cols = params['encoding']['label_encode_cols']
    for col in label_cols:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])
    
    te_cols = params['encoding']['target_encode_cols']
    if te_cols:
        df[te_cols] = target_encoder.transform(df[te_cols])
    
    return df


@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main prediction form"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if all([model, label_encoders, target_encoder]) else "unhealthy"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: hospitalreadmissionriskInput):
    """API endpoint for predictions (JSON)"""
    if not all([model, label_encoders, target_encoder, params]):
        raise HTTPException(status_code=503, detail="Service unavailable")
    try:
        df = engineer_features(input_data.model_dump())

        df_encoded = apply_encodings(df)

        prediction_proba = model.predict_proba(df_encoded)[0, 1]
        threshold = params['model_training']['optimal_threshold']
        prediction = int(prediction_proba >= threshold)


        if prediction_proba >=0.7:
            risk_level = "very high"
        elif prediction_proba >=threshold:
            risk_level = "high"
        elif prediction_proba >= 0.3:
            risk_level = "moderate"
        else:
            risk_level="low"

        return PredictionResponse(
            readmission_risk="High Risk" if prediction == 1 else "Low Risk",
            probability=round(prediction_proba, 4),
            risk_percentage=f"{round(prediction_proba * 100, 2)}%",
            threshold_used=threshold,
            risk_level=risk_level
        )
    

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed {str(e)}")


@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    gender: str = Form(...),
    age: str = Form(...),
    admission_type_id: str = Form(...),
    discharge_disposition_id: str = Form(...),
    admission_source_id: str = Form(...),
    time_in_hospital: int = Form(...),
    num_lab_procedures: int = Form(...),
    num_medications: int = Form(...),
    number_emergency: int = Form(...),
    number_inpatient: int = Form(...),
    number_diagnoses: int = Form(...),
    metformin: str = Form(...),
    change: str = Form(...),
    diabetesmed: str = Form(...),
    diag_1_group: str = Form(...),
    num_med_changes: int = Form(...),
    total_visits: int = Form(...),
    insulin_coded: int = Form(...),
    num_med_active: int = Form(...)
):
    """Form submission endpoint that returns rendered HTML with results"""
    if not all([model, label_encoders, target_encoder, params]):
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "error": "Service unavailable - models not loaded"
            }
        )
    
    try:
        # Create input data
        input_dict = {
            "gender": gender,
            "age": age,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_medications": num_medications,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "number_diagnoses": number_diagnoses,
            "metformin": metformin,
            "change": change,
            "diabetesmed": diabetesmed,
            "diag_1_group": diag_1_group,
            "num_med_changes": num_med_changes,
            "total_visits": total_visits,
            "insulin_coded": insulin_coded,
            "num_med_active": num_med_active
        }
        
        df = engineer_features(input_dict)
        df_encoded = apply_encodings(df)
        
        prediction_proba = model.predict_proba(df_encoded)[0, 1]
        threshold = params['model_training']['optimal_threshold']
        prediction = int(prediction_proba >= threshold)

        if prediction_proba >= 0.7:
            risk_level = "very high"
        elif prediction_proba >= threshold:
            risk_level = "high"
        elif prediction_proba >= 0.3:
            risk_level = "moderate"
        else:
            risk_level = "low"

        result = {
            "readmission_risk": "High Risk" if prediction == 1 else "Low Risk",
            "probability": round(prediction_proba, 4),
            "risk_percentage": f"{round(prediction_proba * 100, 2)}%",
            "threshold_used": round(threshold, 4),
            "risk_level": risk_level.upper(),
            "is_high_risk": prediction == 1
        }
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": result,
                "form_data": input_dict
            }
        )
        
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Prediction failed: {str(e)}"
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)