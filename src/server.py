from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent

from src.inference import EnergyPredictor, get_predictor

predictor = None

def get_loaded_predictor():
    global predictor
    if predictor is None or predictor.loaded == False:
        try:
            predictor = get_predictor()
        except Exception as e:
            print(f"Model load notice: {e}")
    return predictor

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        predictor = get_loaded_predictor()
        print("Model loaded successfully at API startup.")
    except Exception as e:
        print(f"Warning: Could not load model at startup: {e}")
        predictor = None
    yield

app = FastAPI(
    title="Energy Consumption Forecasting API",
    description="Predicts the next-hour household Global Active Power (kW) using 168 hours of historical consumption data.",
    version="1.0.0",
    lifespan=lifespan,
)

class Observation(BaseModel):
    datetime: str = Field(..., description="ISO 8601 timestamp")
    Global_active_power: float = Field(..., ge=0, description="Global active power in kW")
    Global_reactive_power: float = Field(..., ge=0, description="Global reactive power in kW")
    Voltage: float = Field(..., gt=0, description="Voltage in volts")
    Global_intensity: float = Field(..., ge=0, description="Global current intensity in amps")
    Sub_metering_1: float = Field(..., ge=0, description="Sub-metering 1")
    Sub_metering_2: float = Field(..., ge=0, description="Sub-metering 2")
    Sub_metering_3: float = Field(..., ge=0, description="Sub-metering 3")

class PredictionRequest(BaseModel):
    observations: List[Observation] = Field(
        ...,
        min_length=168,
        description="At least 168 hourly observations in chronological order",
    )

class PredictionResponse(BaseModel):
    predicted_consumption_kw: float = Field(..., description="Predicted next-hour Global Active Power in kW")
    model: str = Field(..., description="Name of the model used")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    model_name: str

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    pred = get_loaded_predictor()
    is_ready = pred is not None and pred.loaded
    return HealthResponse(
        status="healthy" if is_ready else "unhealthy",
        model_loaded=is_ready,
        model_name=pred.model_name if is_ready else "Not loaded",
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    pred = get_loaded_predictor()
    if pred is None or pred.loaded == False:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later.",
        )

    try:
        observations = [obs.model_dump() for obs in request.observations]
        prediction = pred.predict(observations)

        return PredictionResponse(
            predicted_consumption_kw=prediction,
            model=pred.model_name,
            timestamp=datetime.now().isoformat(),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)
