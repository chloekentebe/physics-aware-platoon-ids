'''
Purpose timestamp alignment utilities used by sychronization.py
Methods available:
    interpolate_to_grid - primary method: linear for numeric, ffill for categorical
    nearest_merge       - merge_asof wrapper for nearest-timestamp joins
    make_time_grid      - build a clean evenly-spaced grid
'''

import numpy as np
import pandas as pd

def make_time_grid(start: float, stop: float, step: float) -> np.ndarray:
    '''
    Build a clean evenly-spaced time grid from start to stop inclusive.
    Rouds to avoid floating point accumulation drift.
    Example: make_time_grid(0, 60, 0.02) -> [0.00, 0.02, 0.04, ..., 60.00]
    '''
    # use linspace instead of arrange for floating point safety
    n_steps = int(round((stop - start) / step)) + 1
    grid = np.linspace(start, stop, n_steps)
    # round to suppress residual float noise
    decimals = max(0, -int(np.floor(np.log10(step))) + 1)

    return np.round(grid, decimals)

def interpolate_to_grid(
        df: pd.Dataframe,
        time_col: str,
        grid: np.darray,
        group_cols: list[str] | None = None,
        ) -> pd.DataFrame:
    '''
    Interpolates a DataFrame onto a target time grid.
    For each group (or the whole Dataframe if group_cols is empty):
        - Numeric columns:   linear interpolation (method='index')
        - Non-numeric columns: forward fill then backward fill
    
    ***Parameters***
    df          : input DataFrame with a time column
    time_col    : name of the time column
    grid        : target time values to interpolate onto
    group_cols  : columns to group by before interpolating (e.g. ['GlobalRunID])

    ***Returns***
    DataFrame with exactly len(grid) rows per group indexed by time_col
    '''

    group_cols = group_cols or []

    def _interpolate_one_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(time_col).drop_duplicates(subset=[time_col])
        group = group.set_index(time_col)

        # union of existing timestamps and target grid
        full_index = group.index.union(grid).sort_values()
        aligned = group.reindex(full_index)

        # interpolate numeric columns linearly by index position
        numeric_cols = aligned.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            aligned[numeric_cols] = aligned[numeric_cols].interpolate(method="index")
        
        # forward/backward fill categorical and string columns
        other_cols = [col for col in aligned.columns if col not in numeric_cols]
        if other_cols:
            aligned[other_cols] = aligned[other_cols].ffill().bfill()

        # keep only the target grid rows
        return aligned.loc[grid].reset_index().rename(columns={"index": time_col})
    
    if not group_cols:
        return _interpolate_one_group(df)
    
    frames = []
    for _, group in df.groupby(group_cols, dropna=False):
        frames.append(_interpolate_one_group(group))
    
    return pd.concat(frames, ignore_index=True)

def nearest_merge(
        left: pd.DataFrame,
        right: pd.DatFrame, 
        on: str,
        by: list[str] | None = None,
        tolerance: float | None = None,
        direction: str = "nearest") -> pd.DataFrame:
    '''
    Merge two DataFrames on a time column using nearest-timestamp matching.
    Thin wrapped around pd.merge_asof for use in sychronization.

    ***Parameters***
    left, right : DataFrames to merge (both sorted by 'on' before merging)
    on          : time column name
    by          : additional columns that must match exactly (e.g. ['GlobalRunID])
    tolerance   : maximum allowed time gap for a match (None = no limit)
    direction   : 'nearest', 'forward', or 'backward'
    '''
    return pd.merge_asof(
        left.sort_values(on),
        right.sort_values(on),
        on=on,
        by=by,
        tolerance=tolerance,
        direction=direction,
    )