import os
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
application = FastAPI()

with open('model.pkl', 'rb') as file:
    classifier = pickle.load(file)

class TitanicInput(BaseModel):
    pclass: int
    sex: int
    age: float
    fare: float
    family_size: int
    embarked_Q: int
    embarked_S: int

@application.get("/welcome")
async def home_page():
    return {"greeting": "Hello! Welcome to Titanic Prediction API!"}

@application.post("/classify")
async def make_prediction(input_data: TitanicInput):
    try:
        features = np.array([[
            input_data.pclass,
            input_data.sex,
            input_data.age,
            input_data.fare,
            input_data.family_size,
            input_data.embarked_Q,
            input_data.embarked_S
        ]])
        result = classifier.predict(features)[0]
        return {"survival_prediction": int(result)}
    except Exception as error:
        raise HTTPException(status_code=500, detail="Internal server error occurred")

@application.get("/model-details")
async def get_model_details():
    return {
        "model": "Titanic Survival Predictor",
        "feature_count": 7,
        "features": ["pclass", "sex", "age", "fare", "family_size", "embarked_Q", "embarked_S"]
    }

@application.get("/status")
async def check_service_status():
    return {"health": "operational"}
    
@application.get("/config")
async def get_config():
    api_key = os.getenv("API_KEY", "not_set")
    return {"api_key": api_key}
