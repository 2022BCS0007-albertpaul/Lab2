from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Your details
YOUR_NAME = "Albert Sebastian"
ROLL_NO = "2022BCS0007"

# Initialize FastAPI app
app = FastAPI(title="Wine Quality Prediction API")

# Load trained model
model = joblib.load("app/model.pkl")

# Home route (optional)
@app.get("/")
def home():
    return {"message": "Wine Quality Prediction API is running"}

# Input schema
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# Prediction endpoint
@app.post("/predict")
def predict_quality(features: WineFeatures):
    # Prepare input
    data = np.array([[
        features.fixed_acidity,
        features.volatile_acidity,
        features.citric_acid,
        features.residual_sugar,
        features.chlorides,
        features.free_sulfur_dioxide,
        features.total_sulfur_dioxide,
        features.density,
        features.pH,
        features.sulphates,
        features.alcohol
    ]])

    # Make prediction
    prediction = model.predict(data)
    predicted_quality = round(float(prediction[0]), 2)

    # Return in required format
    return {
        "name": "Albert Paul Sebastian",
        "roll_no": "2022BCS0007",
        "wine_quality": predicted_quality
    }
