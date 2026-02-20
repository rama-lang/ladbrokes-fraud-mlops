from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. FastAPI App ని ప్రారంభించడం
app = FastAPI(title="Ladbrokes Betting Fraud Detection API")

# 2. మోడల్ పాత్ సెట్ చేయడం
# Docker లో ఫైల్స్ ఎక్కడ ఉంటాయో ఆ పాత్ ఇవ్వాలి
MODEL_PATH = "models/betting_fraud_model.pkl"

# మోడల్ లోడ్ చేయడం
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# 3. రిక్వెస్ట్ బాడీ కోసం స్కీమా (Data validation)
class BetData(BaseModel):
    bet_amount: float
    bet_count_1h: float
    is_high_stake: int

@app.get("/")
def read_root():
    return {"message": "Ladbrokes Fraud Detection API is Live!", "status": "Ready"}

@app.post("/predict")
def predict_fraud(data: BetData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")
    
    try:
        # వచ్చిన డేటాను DataFrame గా మార్చాలి
        input_data = pd.DataFrame([data.dict()])
        
        # మోడల్ ప్రిడిక్షన్
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] # ఫ్రాడ్ అయ్యే అవకాశం (0 to 1)
        
        result = "Fraudulent" if prediction == 1 else "Normal"
        
        return {
            "prediction": result,
            "fraud_probability": round(float(probability), 4),
            "bet_details": data.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))