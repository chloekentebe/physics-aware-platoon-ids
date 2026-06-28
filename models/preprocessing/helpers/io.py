'''purpose: loading and saving files'''

from pathlib import Path
import json
import pandas as pd

def list_csv_files(folder):
    return sorted(Path(folder).glob("*.csv"))

def load_csv(path):
    return pd.read_csv(path)

def save_dataframe(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def load_dataframe(path):
    path = Path(path)
    return pd.read_csv(path)

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)