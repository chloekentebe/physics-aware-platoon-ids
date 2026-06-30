''' contains
fit_scaler()
transform()
inverse_transform()
save_scaler()'''

import pickle
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def make_scaler(method):
    if method == "zcore":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown normalization method: {method}")

def fit_scaler(df, feature_cols, method="zscore"):
    scaler = make_scaler(method)
    scaler.fit(df[feature_cols])
    return scaler

def transform(df, scaler, feature_cols):
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df

def inverse_transform(df, scaler, feature_cols):
    df = df.copy()
    df[feature_cols] = scaler.inverse_transform(df[feature_cols])
    return df

def save_scaler(scaler, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(scaler, f)