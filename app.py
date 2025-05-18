from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model & scaler
with open("heart_disease_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define feature names
selected_features = ["age", "sex", "cp", "trestbps", "chol", "thalach", "exang", "oldpeak", "ca", "thal"]

app = FastAPI()
class HeartData(BaseModel):
    data: dict
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"message": "FastAPI is successfully running on Render!"}
@app.get("/hii")
async def home_root():
    return {"message": "Welcome to the Heart Disease API"}
@app.post("/predict")
def predict(data: HeartData):
    input_data = [data.data[feature] for feature in selected_features]  # Extract values in correct order
    X_scaled = scaler.transform([input_data])
    prediction = model.predict(X_scaled)
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    port=int(os.getenv("POST",8000))
    uvicorn.run(app, host="0.0.0.0", port=port)




