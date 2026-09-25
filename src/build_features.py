import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent.parent

from src.data_loader import load_data
from src.preprocess import preprocess

MODELS_DIR = BASE_DIR / "models"
FEATURE_SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = MODELS_DIR / "target_scaler.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"

LOOKBACK = 168
HORIZON = 1
TARGET = "Global_active_power"
MAX_LAG = 168
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

def add_time_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df

def add_cyclical_features(df):
    df = df.copy()
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df

def add_lag_features(df):
    df = df.copy()
    df["lag_1"] = df[TARGET].shift(1)
    df["lag_24"] = df[TARGET].shift(24)
    df["lag_168"] = df[TARGET].shift(168)
    return df

def add_rolling_features(df):
    df = df.copy()
    shifted = df[TARGET].shift(1)
    df["rolling_mean_24h"] = shifted.rolling(window=24, min_periods=1).mean()
    df["rolling_std_24h"] = shifted.rolling(window=24, min_periods=1).std()
    df["rolling_std_24h"] = df["rolling_std_24h"].fillna(0)
    return df

def engineer_features(df):
    df = add_time_features(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df.dropna()

def split_chronological(df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(
        f"Chronological split - Train: {train_df.shape} ({train_df.index.min()} -> {train_df.index.max()}) | "
        f"Val: {val_df.shape} ({val_df.index.min()} -> {val_df.index.max()}) | "
        f"Test: {test_df.shape} ({test_df.index.min()} -> {test_df.index.max()})"
    )

    return train_df, val_df, test_df

def build_sequences(features, targets, lookback=LOOKBACK):
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback : i])
        y.append(targets[i])
    return np.array(X), np.array(y)

def build_features(df_hourly=None):
    if df_hourly is None:
        df_hourly = preprocess()

    train_raw, val_raw, test_raw = split_chronological(df_hourly)

    buffer_len = MAX_LAG + LOOKBACK
    val_extended = pd.concat([train_raw.iloc[-buffer_len:], val_raw])
    test_extended = pd.concat([val_raw.iloc[-buffer_len:], test_raw])

    train_df = engineer_features(train_raw)
    val_df_full = engineer_features(val_extended)
    test_df_full = engineer_features(test_extended)

    val_df = val_df_full.iloc[-(len(val_raw) + LOOKBACK) :]
    test_df = test_df_full.iloc[-(len(test_raw) + LOOKBACK) :]

    feature_cols = [c for c in train_df.columns if c != TARGET]

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train_df[feature_cols])

    target_scaler = MinMaxScaler()
    target_scaler.fit(train_df[[TARGET]])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
    joblib.dump(target_scaler, TARGET_SCALER_PATH)
    print(f"Saved scalers to: {MODELS_DIR}")

    feature_config = {
        "feature_columns": feature_cols,
        "target_column": TARGET,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
    }
    with open(FEATURE_CONFIG_PATH, "w") as f:
        json.dump(feature_config, f, indent=2)
    print(f"Saved feature config to: {FEATURE_CONFIG_PATH}")

    train_feat = feature_scaler.transform(train_df[feature_cols])
    val_feat = feature_scaler.transform(val_df[feature_cols])
    test_feat = feature_scaler.transform(test_df[feature_cols])

    train_targ = target_scaler.transform(train_df[[TARGET]]).flatten()
    val_targ = target_scaler.transform(val_df[[TARGET]]).flatten()
    test_targ = target_scaler.transform(test_df[[TARGET]]).flatten()

    X_train_seq, y_train_seq = build_sequences(train_feat, train_targ)
    X_val_seq, y_val_seq = build_sequences(val_feat, val_targ)
    X_test_seq, y_test_seq = build_sequences(test_feat, test_targ)

    print(
        f"Sequences - Train: {X_train_seq.shape} | Val: {X_val_seq.shape} | Test: {X_test_seq.shape}"
    )

    X_train_tab = train_feat[LOOKBACK:]
    y_train_tab = train_targ[LOOKBACK:]
    X_val_tab = val_feat[LOOKBACK:]
    y_val_tab = val_targ[LOOKBACK:]
    X_test_tab = test_feat[LOOKBACK:]
    y_test_tab = test_targ[LOOKBACK:]

    return {
        "X_train_seq": X_train_seq,
        "y_train_seq": y_train_seq,
        "X_val_seq": X_val_seq,
        "y_val_seq": y_val_seq,
        "X_test_seq": X_test_seq,
        "y_test_seq": y_test_seq,
        "X_train_tab": X_train_tab,
        "y_train_tab": y_train_tab,
        "X_val_tab": X_val_tab,
        "y_val_tab": y_val_tab,
        "X_test_tab": X_test_tab,
        "y_test_tab": y_test_tab,
        "feature_cols": feature_cols,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "test_raw": test_raw,
    }

if __name__ == "__main__":
    result = build_features()
    print("Feature columns:", result["feature_cols"])
    print("X_train_seq:", result["X_train_seq"].shape)
    print("X_test_tab:", result["X_test_tab"].shape)