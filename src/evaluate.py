import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports" / "figures"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

TARGET = "Global_active_power"
LOOKBACK = 168

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")

def generate_eda_plots(df_hourly):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating EDA plots...")

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df_hourly.index, df_hourly[TARGET], linewidth=0.3, alpha=0.7)
    ax.set_title("Global Active Power Over Time (Hourly)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Global Active Power (kW)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "consumption_over_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    hourly_avg = df_hourly.groupby(df_hourly.index.hour)[TARGET].mean()
    ax.bar(hourly_avg.index, hourly_avg.values, color="steelblue", edgecolor="white")
    ax.set_title("Average Hourly Consumption Pattern", fontsize=14)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Global Active Power (kW)")
    ax.set_xticks(range(0, 24))
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "daily_pattern.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly_avg = df_hourly.groupby(df_hourly.index.dayofweek)[TARGET].mean()
    ax.bar(range(7), weekly_avg.values, color="coral", edgecolor="white")
    ax.set_title("Average Consumption by Day of Week", fontsize=14)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Mean Global Active Power (kW)")
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "weekly_pattern.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_avg = df_hourly.groupby(df_hourly.index.month)[TARGET].mean()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.bar(range(1, 13), monthly_avg.values, color="seagreen", edgecolor="white")
    ax.set_title("Average Consumption by Month", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Global Active Power (kW)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "monthly_pattern.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (label, mask) in zip(axes, [
        ("Weekday", df_hourly.index.dayofweek < 5),
        ("Weekend", df_hourly.index.dayofweek >= 5),
    ]):
        subset = df_hourly.loc[mask]
        hourly_pattern = subset.groupby(subset.index.hour)[TARGET].mean()
        ax.bar(hourly_pattern.index, hourly_pattern.values, edgecolor="white")
        ax.set_title(f"{label} Hourly Pattern", fontsize=13)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Mean Global Active Power (kW)")
        ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "weekday_vs_weekend.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_hourly[TARGET].dropna(), bins=80, edgecolor="white", alpha=0.8)
    ax.set_title("Distribution of Global Active Power", fontsize=14)
    ax.set_xlabel("Global Active Power (kW)")
    ax.set_ylabel("Frequency")
    ax.axvline(df_hourly[TARGET].mean(), color="red", linestyle="--",
               label=f"Mean: {df_hourly[TARGET].mean():.2f} kW")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df_hourly.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True)
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    missing_pct = (df_hourly.isnull().sum() / len(df_hourly) * 100)
    ax.barh(missing_pct.index, missing_pct.values, color="salmon")
    ax.set_title("Missing Data Percentage by Column (After Processing)", fontsize=14)
    ax.set_xlabel("Missing %")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "missing_data.png", dpi=150)
    plt.close(fig)

    print(f"EDA plots saved to: {REPORTS_DIR}")

def print_comparison_table(results):
    rows = []
    for model_name, res in results.items():
        m = res["metrics"]
        rows.append({
            "Model": model_name,
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "R2": m["R2"],
        })

    df_table = pd.DataFrame(rows).sort_values("MAE")

    print("\n" + "=" * 65)
    print("MODEL COMPARISON (Test Set)")
    print("=" * 65)
    print(df_table.to_string(index=False))
    print("=" * 65 + "\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_table.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    return df_table

def generate_error_analysis(results, data, best_model_name):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best_result = results[best_model_name]
    y_true = best_result["y_test_orig"]
    y_pred = best_result["y_pred_orig"]
    errors = y_true - y_pred

    test_df = data["test_df"]
    test_timestamps = test_df.index[LOOKBACK:]
    min_len = min(len(test_timestamps), len(y_true))
    test_timestamps = test_timestamps[:min_len]
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    errors = errors[:min_len]

    fig, ax = plt.subplots(figsize=(16, 5))
    n_show = min(500, len(y_true))
    ax.plot(test_timestamps[-n_show:], y_true[-n_show:],
            label="Actual", linewidth=1, alpha=0.8)
    ax.plot(test_timestamps[-n_show:], y_pred[-n_show:],
            label="Predicted", linewidth=1, alpha=0.8)
    ax.set_title(f"Actual vs Predicted - {best_model_name} (Last {n_show}h)", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Global Active Power (kW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.1, s=5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect Prediction")
    ax.set_title(f"Actual vs Predicted Scatter - {best_model_name}", fontsize=14)
    ax.set_xlabel("Actual (kW)")
    ax.set_ylabel("Predicted (kW)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "actual_vs_predicted_scatter.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    error_df = pd.DataFrame({
        "error": np.abs(errors),
        "hour": test_timestamps.hour,
    })
    hourly_mae = error_df.groupby("hour")["error"].mean()
    ax.bar(hourly_mae.index, hourly_mae.values, color="steelblue", edgecolor="white")
    ax.set_title(f"MAE by Hour of Day - {best_model_name}", fontsize=14)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("MAE (kW)")
    ax.set_xticks(range(0, 24))
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "error_by_hour.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    error_df["day_of_week"] = test_timestamps.dayofweek
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily_mae = error_df.groupby("day_of_week")["error"].mean()
    ax.bar(range(7), daily_mae.values, color="coral", edgecolor="white")
    ax.set_title(f"MAE by Day of Week - {best_model_name}", fontsize=14)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("MAE (kW)")
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "error_by_day.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    is_weekend = test_timestamps.dayofweek >= 5
    ax.hist(np.abs(errors[~is_weekend]), bins=50, alpha=0.6, label="Weekday", density=True)
    ax.hist(np.abs(errors[is_weekend]), bins=50, alpha=0.6, label="Weekend", density=True)
    ax.set_title(f"Error Distribution: Weekday vs Weekend - {best_model_name}", fontsize=14)
    ax.set_xlabel("Absolute Error (kW)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "error_weekday_weekend.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=80, edgecolor="white", alpha=0.8, color="steelblue")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title(f"Prediction Error Distribution - {best_model_name}", fontsize=14)
    ax.set_xlabel("Error (Actual - Predicted) (kW)")
    ax.set_ylabel("Frequency")
    ax.annotate(f"Mean Error: {errors.mean():.4f}\nStd: {errors.std():.4f}",
                xy=(0.02, 0.95), xycoords="axes fraction", fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "error_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 5))
    rolling_mae = pd.Series(np.abs(errors), index=test_timestamps).rolling(24).mean()
    ax.plot(rolling_mae.index, rolling_mae.values, linewidth=0.8)
    ax.set_title(f"Rolling 24h MAE Over Time - {best_model_name}", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Rolling MAE (kW)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "error_over_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    median_power = np.median(y_true)
    high_mask = y_true >= median_power
    low_mask = y_true < median_power
    categories = ["Low Consumption\n(< median)", "High Consumption\n(>= median)"]
    maes = [float(np.mean(np.abs(errors[low_mask]))), float(np.mean(np.abs(errors[high_mask])))]
    colors = ["lightblue", "salmon"]
    ax.bar(categories, maes, color=colors, edgecolor="white", width=0.5)
    ax.set_title(f"MAE: Low vs High Consumption - {best_model_name}", fontsize=14)
    ax.set_ylabel("MAE (kW)")
    for i, v in enumerate(maes):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "high_low_consumption.png", dpi=150)
    plt.close(fig)

    print(f"Error analysis plots saved to: {REPORTS_DIR}")

def generate_training_loss_plots(results):
    tf_models = ["MLP", "LSTM", "GRU"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, model_name in zip(axes, tf_models):
        if model_name in results and "history" in results[model_name]:
            hist = results[model_name]["history"]
            ax.plot(hist["loss"], label="Train Loss")
            ax.plot(hist["val_loss"], label="Val Loss")
            ax.set_title(f"{model_name} - Training History", fontsize=13)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Loss")
            ax.legend()
        else:
            ax.set_title(f"{model_name} - No history", fontsize=13)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "training_loss.png", dpi=150)
    plt.close(fig)

    print(f"Training loss plots saved to: {REPORTS_DIR / 'training_loss.png'}")

def select_best_model(results):
    best_name = None
    best_mae = float("inf")

    for name, res in results.items():
        mae = res["metrics"]["MAE"]
        if mae < best_mae:
            best_mae = mae
            best_name = name

    print(f"Selected best model: {best_name} (MAE={best_mae:.4f})")
    return best_name

def save_best_model(results, best_model_name):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tf_models = {"MLP": "mlp", "LSTM": "lstm", "GRU": "gru"}

    if best_model_name in tf_models:
        source_path = MODELS_DIR / f"{tf_models[best_model_name]}.keras"
        if source_path.exists():
            shutil.copy2(source_path, BEST_MODEL_PATH)
            print(f"Saved best TF model to: {BEST_MODEL_PATH}")
        model_type = "tensorflow"
    else:
        model_type = "sklearn"

    metadata = {
        "model_name": best_model_name,
        "model_type": model_type,
        "metrics": results[best_model_name]["metrics"],
        "all_model_metrics": {
            name: res["metrics"] for name, res in results.items()
        },
        "selection_criteria": "Lowest test MAE among all models",
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model metadata to: {MODEL_METADATA_PATH}")

def run_evaluation(results, data):
    comparison_df = print_comparison_table(results)
    generate_training_loss_plots(results)
    best_model_name = select_best_model(results)
    generate_error_analysis(results, data, best_model_name)
    save_best_model(results, best_model_name)
    return best_model_name

if __name__ == "__main__":
    print("Evaluation module loaded.")
