from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'household_power_consumption.csv'


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, na_values=['?'], low_memory=False)

    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df = df.drop(columns=['Date', 'Time']).set_index('datetime').sort_index()

    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} > {df.index.max()}")
    print(f"Missing values before fill:\n{df.isnull().sum()}\n")
    print(f"Duplicate timestamps: {df.index.duplicated().sum()}")

    return df

if __name__ == '__main__':
    df = load_data()
    print(df.head())