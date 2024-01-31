# Import necessary modules from FastAPI, Pydantic, joblib, and pandas

from fastapi import FastAPI
from pydantic import BaseModel 
import joblib
import pandas as pd 

# Define a Pydantic model that represents the data expected in the request body

class ScoringItem(BaseModel):
    YearMade: float
    sale_year: float
    sale_month: float
    sale_day: float
    sale_dayofweek: float
    fiModelDesc_label_encoded: float
    fiBaseModel_label_encoded: float
    state_label_encoded: float


# Load the pre-trained machine learning model using joblib
with open("trained_model.pkl", "rb") as f:
    model = joblib.load(f)

# Define a FastAPI app
app = FastAPI()

# Define a route using the @app.post decorator to handle POST requests
@app.post('/')
async def scoring_endpoint(item:ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns = item.dict().keys())
    yhat = model.predict(df)
    return {"Predication": int(yhat)}
    