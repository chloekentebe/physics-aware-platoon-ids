import numpy as np

def require_columns(df, columns, context="dataframe"):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{context} missing columns: {missing}")

def require_non_empty(df, context="dataframe"):
    if df.empty:
        raise ValueError(f"{context} is empty")
    
def require_monotonic(df, time_col, group_cols=None, context="dataframe"):
    group_cols = group_cols or []
    groups = df.groupby(group_cols, dropna=False) if group_cols else [(None, df)]

    for key, group in groups:
        values = group[time_col].to_numpy()
        if np.any(np.diff(values) < 0):
            raise ValueError(f"{context} has non-monotonic timestamps in group {key}")
        
def report_quality(df):
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isna.sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }