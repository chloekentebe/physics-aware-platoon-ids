'''Purpose: quality control checks called throughout the pipeline.
Every check raises ValueError on failure or returns a quality report dict.
'''

import numpy as np
import pandas as pd

def require_columns(df: pd.DataFrame, columns: list[str], context: str = "dataframe"):
    '''raises if any required column in missing'''
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{context} missing columns: {missing}")

def require_non_empty(df: pd.DataFrame, context: str = "dataframe"):
    '''raises if the DataFrame has no rows'''
    if df.empty:
        raise ValueError(f"{context} is empty")
    
def require_monotonic(
    df: pd.DataFrame,
    time_col: str,
    group_cols: list[str] | None = None,
    context: str = "dataframe",
):
    '''raises ValueError if timesteps are non-monotonic with any gorup
       group cols should be the channel identifier, e.g. ['ReceiverID', 'SenderSlot']
    '''
    group_cols = group_cols or []
    groups = df.groupby(group_cols, dropna=False) if group_cols else [(None, df)]

    for key, group in groups:
        values = group[time_col].to_numpy()
        if np.any(np.diff(values) < 0):
            raise ValueError(f"{context} has non-monotonic timestamps in group {key}")
        
def check_speed_range(
    df: pd.DataFrame,
    speed_col: str,
    min_speed: float = -1.0,
    max_speed: float = 50.0,
    context: str = "dataframe",
) -> dict:
    '''returns a report out-of-range speed values (does not raise)'''
    if speed_col not in df.columns:
        return {"checked": False, "reason": f"{speed_col} not in DataFrame"}
    out_of_range = df[(df[speed_col] < min_speed) | (df[speed_col] > max_speed)]
    return {
        "column": speed_col,
        "n_out_of_range": int(len(out_of_range)),
        "fraction": float(len(out_of_range) / max(len(df), 1)),
        "min_observed": float(df[speed_col].min()),
        "max_observed": float(df[speed_col].max()),
    }

def check_spacing_physical(
    df: pd.DataFrame,
    spacing_cols: list[str],
    min_spacing: float = 0.5,
    context: str = "dataframe",
) -> dict:
    '''returns a report of physically impossible spacings'''
    report = {}
    for col in spacing_cols:
            if col not in df.columns:
                continue
            n_impossible = int((df[col] <= min_spacing).sum())  
            report[col] = {
                "n_impossible": n_impossible,
                "fraction": float(n_impossible / max(len(df), 1)),
                "min_observed": float(df[col].min()),
            }
    return report

def check_msgcnt_continuity(
    df: pd.DataFrame,
    msgcnt_col: str = "bsm_msg_count",
    group_cols: list[str] | None = None,
    context: str = "dataframe",
) -> dict:
    '''
    checks that MsgCnt increments by exactly 1 each step (wrapping at 128)
    returns fraction of steps with unexpected increments — non-zero in attack runs
    '''
    if msgcnt_col not in df.columns:
        return {"checked": False}
    group_cols = group_cols or []
    groups = df.groupby(group_cols, dropna=False) if group_cols else [(None, df)]

    total, anomalous = 0, 0
    for _, group in groups:
        cnt = group[msgcnt_col].to_numpy()
        delta = np.diff(cnt)
        # Account for wrap: 127 -> 0 is a valid +1 step
        delta_wrapped = np.where(delta < -64, delta + 128, delta)
        total += len(delta_wrapped)
        anomalous += int(np.sum(delta_wrapped != 1))

    return {
        "total_steps": total,
        "anomalous_steps": anomalous,
        "anomaly_fraction": float(anomalous / max(total, 1)),
    }

        
def report_quality(df: pd.DataFrame) -> dict:
   '''returns a summary quality report for any DataFrame'''
   return {
        "rows":             int(len(df)),
        "columns":          int(len(df.columns)),
        "missing_values":   int(df.isna().sum().sum()),  
        "duplicate_rows":   int(df.duplicated().sum()),
        "memory_mb":        round(df.memory_usage(deep=True).sum() / 1e6, 2),
        }