'''purpose: loading and saving files'''

from pathlib import Path
import json
import pandas as pd

def list_csv_files(folder):
    return sorted(Path(folder).glob("*.csv"))

def load_csv(path):
    return pd.read_csv(path)

def save_dataframe(df, path):
    '''
    saves DataFrame to parquet or CSV based on file extension
    
    parquet is: 5-10x smaller, preserves dtypes,
    reads 10x faster than CSV for large DataFrames.
    '''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_dataframe(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)