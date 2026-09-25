import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent

from src.data_loader import load_data
from src.preprocess import preprocess, PROCESSED_PARQUET_PATH
from src.build_features import build_features
from src.evaluate import run_evaluation, generate_eda_plots

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports" / "figures"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

SEED = 42
BATCH_SIZE = 128
EPOCHS = 40
PATIENCE_EARLY_STOP = 7
PATIENCE_LR = 3
LR = 1e-3
LR_FACTOR = 0.5
DROPOUT_RATE = 0.2
LOOKBACK = 168

np.random.seed(SEED)
tf.random.set_seed(SEED)

def compute_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

def inverse_scale_predictions(y_scaled, target_scaler):
    return target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

def train_naive_baseline(data):
    y_test = data["y_test_tab"]
    target_scaler = data["target_scaler"]

    y_test_orig = inverse_scale_predictions(y_test, target_scaler)
    y_pred_orig = np.roll(y_test_orig, 1)
    y_pred_orig[0] = y_test_orig[0]

    metrics = compute_metrics(y_test_orig[1:], y_pred_orig[1:])
    print(f"Naive Baseline -> {metrics}")
    return {
        "metrics": metrics,
        "y_test_orig": y_test_orig,
        "y_pred_orig": y_pred_orig,
    }

def train_linear_regression(data):
    target_scaler = data["target_scaler"]
    y_test_orig = inverse_scale_predictions(data["y_test_tab"], target_scaler)
    model_path = MODELS_DIR / "linear_regression.pkl"

    if model_path.exists():
        model = joblib.load(model_path)
    else:
        model = LinearRegression()
        model.fit(data["X_train_tab"], data["y_train_tab"])
        joblib.dump(model, model_path)

    y_pred_scaled = model.predict(data["X_test_tab"])
    y_pred_orig = inverse_scale_predictions(y_pred_scaled, target_scaler)

    metrics = compute_metrics(y_test_orig, y_pred_orig)
    print(f"Linear Regression -> {metrics}")
    return {
        "metrics": metrics,
        "y_test_orig": y_test_orig,
        "y_pred_orig": y_pred_orig,
    }

def train_random_forest(data):
    target_scaler = data["target_scaler"]
    y_test_orig = inverse_scale_predictions(data["y_test_tab"], target_scaler)
    model_path = MODELS_DIR / "random_forest.pkl"

    if model_path.exists():
        model = joblib.load(model_path)
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=SEED,
        )
        model.fit(data["X_train_tab"], data["y_train_tab"])
        joblib.dump(model, model_path)

    y_pred_scaled = model.predict(data["X_test_tab"])
    y_pred_orig = inverse_scale_predictions(y_pred_scaled, target_scaler)

    metrics = compute_metrics(y_test_orig, y_pred_orig)
    print(f"Random Forest -> {metrics}")
    return {
        "metrics": metrics,
        "y_test_orig": y_test_orig,
        "y_pred_orig": y_pred_orig,
    }

def get_callbacks(model_name):
    checkpoint_path = MODELS_DIR / f"{model_name}.keras"
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_EARLY_STOP,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=PATIENCE_LR,
            factor=LR_FACTOR,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

def build_mlp(n_features):
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ], name="MLP")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="mse",
    )
    return model

def build_lstm(lookback, n_features):
    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, n_features)),
        keras.layers.LSTM(32),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ], name="LSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="mse",
    )
    return model

def build_gru(lookback, n_features):
    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, n_features)),
        keras.layers.GRU(32),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ], name="GRU")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="mse",
    )
    return model

def train_mlp(data):
    target_scaler = data["target_scaler"]
    model_path = MODELS_DIR / "mlp.keras"

    if model_path.exists():
        print(f"Loading saved MLP from: {model_path}")
        model = keras.models.load_model(str(model_path))
    else:
        print("\n--- Training MLP ---")
        model = build_mlp(data["X_train_tab"].shape[1])
        model.summary()
        callbacks = get_callbacks("mlp")
        model.fit(
            data["X_train_tab"],
            data["y_train_tab"],
            validation_data=(data["X_val_tab"], data["y_val_tab"]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

    y_pred_scaled = model.predict(data["X_test_tab"], batch_size=BATCH_SIZE).flatten()
    y_test_orig = inverse_scale_predictions(data["y_test_tab"], target_scaler)
    y_pred_orig = inverse_scale_predictions(y_pred_scaled, target_scaler)

    metrics = compute_metrics(y_test_orig, y_pred_orig)
    print(f"MLP -> {metrics}")
    return {
        "metrics": metrics,
        "history": {"loss": [0], "val_loss": [0]},
        "y_test_orig": y_test_orig,
        "y_pred_orig": y_pred_orig,
        "model": model,
    }

def train_gru(data):
    target_scaler = data["target_scaler"]
    model_path = MODELS_DIR / "gru.keras"

    if model_path.exists():
        print(f"Loading saved GRU from: {model_path}")
        model = keras.models.load_model(str(model_path))
    else:
        print("\n--- Training GRU ---")
        model = build_gru(LOOKBACK, data["X_train_seq"].shape[2])
        model.summary()
        callbacks = get_callbacks("gru")
        model.fit(
            data["X_train_seq"],
            data["y_train_seq"],
            validation_data=(data["X_val_seq"], data["y_val_seq"]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

    y_pred_scaled = model.predict(data["X_test_seq"], batch_size=BATCH_SIZE).flatten()
    y_test_orig = inverse_scale_predictions(data["y_test_seq"], target_scaler)
    y_pred_orig = inverse_scale_predictions(y_pred_scaled, target_scaler)

    metrics = compute_metrics(y_test_orig, y_pred_orig)
    print(f"GRU -> {metrics}")
    return {
        "metrics": metrics,
        "history": {"loss": [0], "val_loss": [0]},
        "y_test_orig": y_test_orig,
        "y_pred_orig": y_pred_orig,
        "model": model,
    }

def train_lstm(data):
    target_scaler = data["target_scaler"]
    model_path = MODELS_DIR / "lstm.keras"

    if model_path.exists():
        print(f"Loading saved LSTM from: {model_path}")
        model = keras.models.load_model(str(model_path))
    else:
        print("\n--- Training LSTM ---")
        model = build_lstm(LOOKBACK, data["X_train_seq"].shape[2])
        model.summary()
        callbacks = get_callbacks("lstm")
        model.fit(
            data["X_train_seq"],
            data["y_train_seq"],
            validation_data=(data["X_val_seq"], data["y_val_seq"]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

    y_pred_scaled = model.predict(data["X_test_seq"], batch_size=BATCH_SIZE).flatten()
    y_test_orig = inverse_scale_predictions(data["y_test_seq"], target_scaler)
    y_pred_orig = inverse_scale_predictions(y_pred_scaled, target_scaler)

    metrics = compute_metrics(y_test_orig, y_pred_orig)
    print(f"LSTM -> {metrics}")
    return {
        "metrics": metrics,
        "history": {"loss": [0], "val_loss": [0]},
        "y_test_orig": y_test_orig,
        "y_pred_orig": y_pred_orig,
        "model": model,
    }

def train_all_models(data=None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if data is None:
        if PROCESSED_PARQUET_PATH.exists():
            import pandas as pd
            print(f"Loading cached hourly data from: {PROCESSED_PARQUET_PATH}")
            hourly_df = pd.read_parquet(PROCESSED_PARQUET_PATH)
        else:
            raw_df = load_data()
            hourly_df = preprocess(raw_df)

        data = build_features(hourly_df)

    results = {}

    print("\n" + "=" * 60)
    print("TRAINING BASELINE MODELS")
    print("=" * 60)

    results["Naive Persistence"] = train_naive_baseline(data)
    results["Linear Regression"] = train_linear_regression(data)
    results["Random Forest"] = train_random_forest(data)

    print("\n" + "=" * 60)
    print("TRAINING TENSORFLOW MODELS")
    print("=" * 60)

    results["MLP"] = train_mlp(data)
    results["GRU"] = train_gru(data)
    results["LSTM"] = train_lstm(data)

    return results, data

if __name__ == "__main__":
    print("=" * 70)
    print("ENERGY CONSUMPTION FORECASTING - TRAINING PIPELINE")
    print("=" * 70)

    if PROCESSED_PARQUET_PATH.exists():
        import pandas as pd
        print(f"Loading cached hourly data from: {PROCESSED_PARQUET_PATH}")
        hourly_df = pd.read_parquet(PROCESSED_PARQUET_PATH)
    else:
        raw_df = load_data()
        hourly_df = preprocess(raw_df)

    generate_eda_plots(hourly_df)
    data = build_features(hourly_df)
    results, data = train_all_models(data)
    best_model_name = run_evaluation(results, data)

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE - Best Model Selected: {best_model_name}")
    print("=" * 70)
