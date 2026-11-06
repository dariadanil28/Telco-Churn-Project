import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(csv_path):
    return pd.read_csv(csv_path)


def split_data(df):
    X = df.drop(columns=["Churn", "customerID", "TotalCharges"])
    y = df["Churn"].copy()

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )