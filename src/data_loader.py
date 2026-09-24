from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "household_power_consumption.csv"

def load_data():
    print(f"Loading data")

    df = pd.read_csv(DATA_PATH, na_values=["?"], low_memory=False)

    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
    df = df.drop(columns=["Date", "Time"])
    df = df.set_index("datetime").sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_rows = df.isnull().any(axis=1).sum()
    missing_pct = df.isnull().any(axis=1).mean() * 100
    duplicate_timestamps = df.index.duplicated().sum()
    
    print(f"Loaded {len(df):,} rows | {len(df.columns):,} columns")
    print(f"Date range: {df.index.min()} -> {df.index.max()}")
    print(f"Missing: {missing_rows:,} ({missing_pct:.2f}%)")
    print(f"Duplicate timestamps: {duplicate_timestamps:,}")

    return df

df = load_data()

print(df.head())
print(df.info())