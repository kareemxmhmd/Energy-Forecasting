from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

from data_loader import load_data

DATA_PROCESSED_DIR = BASE_DIR / "data"
PROCESSED_PARQUET_PATH = DATA_PROCESSED_DIR / "hourly_power.parquet"

MEAN_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
]

SUM_COLUMNS = [
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

def remove_duplicates(df):
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        print(f"Removing {n_dup} duplicate timestamps")
        df = df[~df.index.duplicated(keep="first")]
    else:
        print("No duplicate timestamps found")
    return df

def resample_hourly(df):
    df = df.copy()
    
    df["active_energy_wh"] = (df["Global_active_power"] * 1000.0) / 60.0

    agg_dict = {}
    for col in df.columns:
        if col in MEAN_COLUMNS:
            agg_dict[col] = "mean"
        elif col in SUM_COLUMNS:
            agg_dict[col] = lambda s: s.sum(min_count=1)
        else:
            agg_dict[col] = "mean"

    df_hourly = df.resample("1h").agg(agg_dict)
    print(f"Resampled to hourly: {len(df_hourly):,} rows")
    return df_hourly

def fill_missing_values(df):
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before fill: {missing_before:,}")

    df = df.interpolate(method="time", limit=2)
    df = df.fillna(df.shift(168)).fillna(df.shift(-168))
    df = df.bfill().ffill()

    missing_after = df.isnull().sum().sum()
    print(f"Missing values after fill: {missing_after:,}")
    return df

def add_submetering_remainder(df):
    sub_meters_sum = (df["Sub_metering_1"] + df["Sub_metering_2"] + df["Sub_metering_3"])
    df["Sub_metering_remainder"] = (df["active_energy_wh"] - sub_meters_sum).clip(lower=0.0)
    return df

def preprocess(df=None, save=True):
    if df is None:
        df = load_data()
 
    df = remove_duplicates(df)
    df = df.sort_index()
    df = resample_hourly(df)
    df = fill_missing_values(df)
    df = add_submetering_remainder(df)
 
    if save:
        df.to_parquet(PROCESSED_PARQUET_PATH)
        print("Saved processed data")
 
    print(
        f"Preprocessing complete: {len(df):,} rows, {len(df.columns)} cols, "
        f"range: {df.index.min()} -> {df.index.max()}"
    )
 
    return df
 
 
result = preprocess()
print(result.head())
print(result.info())
