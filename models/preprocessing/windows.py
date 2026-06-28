'''purpose: construct temporal sequences
this converts rows into time windows

input: single dataframe
output: tensor

each window gets labels (attack, vehicle, attack type, scenario, run)'''

import numpy as np

from config import CONFIG
from helpers.io import load_dataframe, save_json
from helpers.windows import create_windows
from helpers.utils import ensure_dir, log

LABEL_COLUMNS = [
    "is_attack",
    "attack_vector",
    "attack_type",
    "attacker_id",
    "Profile",
    "ScenarioType",
    "dataset",
    "GlobalRunID"
]

def create_temporal_windows(normalized=None, feature_cols=None):
    if normalized is None:
        normalized = load_dataframe(CONFIG.normalized_dir / CONFIG.normalized_file)
    
    if feature_cols is None:
        feature_cols = [
            col for col in normalized.select_dtypes(include="number").columns
            if col not in set(CONFIG.id_columns) and col not in {"time", "Time"}
        ]
    
    label_cols = [col for col in LABEL_COLUMNS if col in normalized.columns]
    group_cols = [
        col for col in ["GlobalRunID", "ReceiverID", "SenderID"]
        if col in normalized.columns
    ]

    log(f"Creating windows: {len(feature_cols)} features, "
        f"window={CONFIG.window_size}, stride={CONFIG.stride}, "
        f"groups={group_cols}")

    X, labels = create_windows(
        normalized,
        feature_cols=feature_cols,
        label_cols = label_cols,
        group_cols=group_cols,
        window_size=CONFIG.window_size,
        stride=CONFIG.stride,
    )
    log(f"Created {len(X):,} windows, shape {X.shape}")
    ensure_dir(CONFIG.windows_dir)

    arrays = {"X": X}
    for key, values in labels.items():
        arrays[f"label_{key}"] = np.asarray(values)
    
    np.savez_compressed(CONFIG.windows_dir / CONFIG.windows_file, **arrays)
    save_json(feature_cols, CONFIG.windows_dir / "window_feature_names.json")
    save_json(label_cols, CONFIG.windows_dir / "window_label_names.json")
    log(f"Saved windows -> {CONFIG.windows_dir / CONFIG.windows_file}")

    return X, labels

if __name__ == "__main__":
    create_temporal_windows()