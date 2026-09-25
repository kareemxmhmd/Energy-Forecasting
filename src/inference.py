import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

from src.build_features import (
    add_time_features,
    add_cyclical_features,
    add_lag_features,
    add_rolling_features,
)

MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
FEATURE_SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = MODELS_DIR / "target_scaler.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

LOOKBACK = 168
TARGET = "Global_active_power"

class EnergyPredictor:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_config = None
        self.metadata = None
        self.model_name = "Not loaded"
        self.loaded = False

    def load(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        if MODEL_METADATA_PATH.exists() == False:
            raise FileNotFoundError(
                f"Model metadata not found at {MODEL_METADATA_PATH}. Run training first."
            )

        with open(MODEL_METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

        model_type = self.metadata.get("model_type", "tensorflow")
        self.model_name = self.metadata.get("model_name", "Unknown")

        if model_type == "tensorflow":
            from tensorflow import keras
            self.model = keras.models.load_model(str(BEST_MODEL_PATH))
            print(f"Loaded TensorFlow model: {self.model_name}")
        else:
            sklearn_name = self.model_name.lower().replace(" ", "_")
            sklearn_path = self.models_dir / f"{sklearn_name}.pkl"
            self.model = joblib.load(sklearn_path)
            print(f"Loaded scikit-learn model: {self.model_name}")

        self.feature_scaler = joblib.load(FEATURE_SCALER_PATH)
        self.target_scaler = joblib.load(TARGET_SCALER_PATH)

        with open(FEATURE_CONFIG_PATH, "r") as f:
            self.feature_config = json.load(f)

        self.loaded = True
        print("Inference pipeline loaded successfully.")

    @property
    def is_loaded(self):
        return self.loaded

    def validate_input(self, observations):
        if observations is None or len(observations) < LOOKBACK:
            raise ValueError(
                f"Need at least {LOOKBACK} hourly observations, got {len(observations) if observations else 0}"
            )

        df = pd.DataFrame(observations)

        if "datetime" not in df.columns:
            raise ValueError("Each observation must have a 'datetime' field")

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()

        required_cols = [
            "Global_active_power",
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[required_cols].isnull().any().any():
            nan_counts = df[required_cols].isnull().sum()
            nan_cols = nan_counts[nan_counts > 0].to_dict()
            raise ValueError(f"NaN values found in columns: {nan_cols}")

        return df

    _validate_input = validate_input

    def predict(self, observations):
        if self.loaded == False:
            raise RuntimeError("Model not loaded. Call load() first.")

        df = self.validate_input(observations)

        df = add_time_features(df)
        df = add_cyclical_features(df)
        df = add_lag_features(df)
        df = add_rolling_features(df)
        df = df.bfill()

        feature_cols = self.feature_config["feature_columns"]
        features = df[feature_cols].values
        features_scaled = self.feature_scaler.transform(features)

        model_type = self.metadata.get("model_type", "tensorflow")

        if self.model_name in ("LSTM", "GRU"):
            sequence = features_scaled[-LOOKBACK:]
            X = sequence.reshape(1, LOOKBACK, -1)
        else:
            X = features_scaled[-1:].reshape(1, -1)

        if model_type == "tensorflow":
            y_pred_scaled = self.model.predict(X, verbose=0).flatten()
        else:
            y_pred_scaled = self.model.predict(X).flatten()

        y_pred = self.target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()[0]

        return round(float(y_pred), 4)

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = EnergyPredictor()
        _predictor.load()
    return _predictor

def predict_next_hour(observations):
    return get_predictor().predict(observations)

if __name__ == "__main__":
    predictor = EnergyPredictor()
    predictor.load()
    print(f"Loaded model: {predictor.model_name}")
    print("Inference pipeline ready.")
