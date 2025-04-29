from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd

app = FastAPI(
    title="World Happiness Predictor",
    description="Predict Happiness Score based on input features",
    version="0.1",
)

class HappinessFeatures(BaseModel):
    GDP_per_Capita: float
    Social_Support: float
    Healthy_Life_Expectancy: float
    Freedom: float
    Generosity: float
    Corruption_Perception: float
    Unemployment_Rate: float
    Education_Index: float
    Population: float
    Urbanization_Rate: float
    Life_Satisfaction: float
    Public_Trust: float
    Mental_Health_Index: float
    Income_Inequality: float
    Public_Health_Expenditure: float
    Climate_Index: float
    Work_Life_Balance: float
    Internet_Access: float
    Crime_Rate: float
    Political_Stability: float
    Employment_Rate: float

@app.on_event('startup')
def load_model():
    global model
    mlflow.set_tracking_uri("http://127.0.0.1:5000")   # ← important fix
    model = mlflow.sklearn.load_model("models:/WorldHappinessModel_v2/Production")

@app.get("/")
def root():
    return {"message": "World Happiness Prediction API is running."}

@app.post("/predict")
def predict(features: HappinessFeatures):
    X = pd.DataFrame([features.dict()])
    prediction = model.predict(X)
    return {"predicted_happiness_score": prediction[0]}