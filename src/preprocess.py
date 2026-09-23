import pandas as pd
from data_loader import load_data

def preprocess():
    df = load_data()

    duplicates = df.index.duplicated().sum()
    print("Duplicates:", duplicates)
    if duplicates > 0:
        df = df[~df.index.duplicated(keep='first')]

    df = df.sort_index()

    df_resampled = df.resample('1h').mean()

    print("Missing hours before fill:")
    print(df_resampled.isnull().sum())

    df_resampled = df_resampled.fillna(df_resampled.shift(168))
    df_resampled = df_resampled.interpolate(method='time').bfill()

    print("Missing hours after fill:")
    print(df_resampled.isnull().sum())

    print("Resampled shape:", df_resampled.shape)

    return df_resampled

df = preprocess()
print(df.head())