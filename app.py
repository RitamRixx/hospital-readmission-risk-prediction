from fastapi import FastAPI, HTTPException
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd
import uvicorn
import pickle

try:
    with open("models/model.pkl", 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Model pickle not found")
    model=None


