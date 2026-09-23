import pandas as pd

try:
    from data_loader import load_data
except:
    from src.data_loader import load_data


def preprocess():
    df = load_data()

    duplicates = df.index.duplicated().sum()
    print("Duplicates:", duplicates)
    if duplicates > 0:
        df = df[~df.index.duplicated(keep='first')]

    df = df.sort_index()

    print("Missing before:")
    print(df.isnull().sum())

    df = df.fillna(0)

    print("Missing after:")
    print(df.isnull().sum())

    df_resampled = df.resample('1h').mean()
    df_resampled = df_resampled.fillna(0)

    print("Resampled shape:", df_resampled.shape)

    return df_resampled


if __name__ == '__main__':
    df = preprocess()
    print(df.head())
