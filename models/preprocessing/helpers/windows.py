'''Purpose: generate sliding windows from a sychronized, feature-engineered DataFrame.

Stride controls overlap:
    window_size=250, stride=125 -> 50% overlap (each window shares 125 steps with next)
    window_size=250, stride=250 -> no overlap (each window is fully independent)
    window_size=250, stride=25 -> 90% overlap (very dense, many windows per run)

The label per window uses these aggregation rules:
    is_attack:      max() -> window is attack if ANY timestep is under attack
    attack_vector:  last() -> categorical, homogenoue within a run
    attacker_id:    last() -> categorical, homogenous within a run
    attack_type:    last() -> categorical, homogenous within a run
'''

import numpy as np

def window_iterator(df, group_cols, window_size, stride):
    '''
    yields (group_key, window_df) for every sliding windows across every group
    groups are typicaly (GlobalRunID, ReceiverID, SenderSlot) for BSM data or
    (GlobalRunID,) for platoon data
    '''
    groups = df.groupby(group_cols, dropna=False) if group_cols else [(None, df)]

    for group_key, group in groups:
        group = group.sort_values("time").reset_index(drop=True)

        for start in range(0, len(group) - window_size + 1, stride):
            end = start + window_size
            yield group_key, group.iloc[start:end]

# label aggregation rules per columns
# max() = window is positive if ANY row is positive (used for binary flags)
# last() = use the last row's value (safe for columns constant within a run)
_LABEL_AGG = {
    "is_attack":        "max",
    "attack_vector":    "last",
    "attacker_id":      "last",
    "attack_type":      "last",
}

def create_windows(df, feature_cols, label_cols, group_cols, window_size, stride):
    '''
    converts a flat DataFrame into sliding window tensors

    ***returns***
    X :      np.ndarray [n_windows, window_size, n_features]
    labels: dict        {col_name:  list of per-window label values}
    '''
    X, labels = [], {col: [] for col in label_cols}

    for _, window in window_iterator(df, group_cols, window_size, stride):
        X.append(window[feature_cols].to_numpy(dtype=np.float32))

        for col in  label_cols:
            agg = _LABEL_AGG.get(col, "last")
            if agg == "max":
                labels[col].append(int(window[col].max()))
            else:
                labels[col].append(str(window[col].iloc[-1]))
    
    return np.asarray(X, dtype=np.float32), labels