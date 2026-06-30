'''Purpose: scale numerical features for the BiLSTM
    Always normalize: spacings, speeds, accelerations, consistency features,
            temporal rolling features, positions, headings
    The fitted sclar is saved to disk so inference uses identical scaling.
    Methods:
        zcore - zero mean, unit variance (gradients are well conditions when
            features are ~N(0,1))
        minmax - scales to [0,1] (use when features have hard physical bounds
                that want to be preserved)
'''

import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config import CONFIG
from helpers.utils import ensure_dir, log

# columns that must never be normalized
# these are identifiers, labels, or categoricals where scaling them
# would corrupt the label arrays passed ot the model

NEVER_NORMALIZE = {
    # run identity
    "GlobalRunID", "RunID", "loop_idx", "source_file", "dataset",
    # time
    "time", "Time",
    # attack labels — every level of the hierarchy
    "is_attack", "attack_vector", "attack_type", "attacker_id",
    "attack_active", "IsSenderAttacker",
    "IsAttack", "AttackVector", "AttackType", "AttackerID", "AttackActive",
    # attack metadata (ranges, not signals)
    "AttackMag", "AttackStart", "AttackDuration",
    "TimingShift1", "TimingShift2", "TimingShift3",
    "AttackDelta1", "AttackDelta2", "AttackDelta3",
    # vehicle / communication identity
    "ReceiverID", "SenderID", "SenderSlot", "bsm_vehicle_id",
    # scenario descriptors
    "Profile", "ScenarioType", "source_type", "table",
    # BSM protocol fields that are enums/IDs not physical measurements
    "bsm_msg_count",        # raw counter —  bsm_msg_count_deviation is used instead
    "bsm_sec_mark",
}

# feature groups (what should be mormalized)
# organized by section from features.py

# SECTION A: raw platoon ground truth signals
PLATOON_FEATURES = [
    "platoon_speed_leader",
    "platoon_speed_follower1", "platoon_speed_follower2",
    "platoon_speed_follower3", "platoon_speed_follower4",
    "platoon_accel_leader",
    "platoon_accel_follower1", "platoon_accel_follower2",
    "platoon_accel_follower3", "platoon_accel_follower4",
    "spacing_l_f1", "spacing_f1_f2", "spacing_f2_f3", "spacing_f3_f4",
]

# SECTION B: BSM communication signals (physical-unit decoded)
BSM_FEATURES = [
    "bsm_speed", "bsm_long_accel", "bsm_lat_accel",
    "bsm_vert_accel", "bsm_yaw_rate", "bsm_heading",
    "bsm_latitude", "bsm_longitude", "bsm_elevation",
    "bsm_derived_accel", "bsm_heading_rate",
    "bsm_msg_count_delta", "bsm_msg_count_deviation",
    "true_speed_ms", "reported_speed_ms", "speed_deviation_raw",
]

# SECTION C: physics consistency features
CONSISTENCY_FEATURES = [
    "speed_consistency_error", "speed_consistency_abs_error",
    "speed_consistency_rel_error",
    "accel_consistency_error", "bsm_internal_accel_error",
    "position_velocity_consistency",
    "bsm_position_displacement", "bsm_expected_displacement",
    "time_headway", "spacing_error_2s", "spacing_error_abs",
    "relative_speed_leader_follower1", "closing_rate_leader_follower1",
    "relative_speed_follower1_follower2", "closing_rate_follower1_follower2",
    "relative_speed_follower2_follower3", "closing_rate_follower2_follower3",
    "relative_speed_follower3_follower4", "closing_rate_follower3_follower4",
    "relative_accel_leader_follower1", "relative_accel_follower1_follower2",
    "relative_accel_follower2_follower3", "relative_accel_follower3_follower4",
    "speed_diff_leader_follower1", "speed_diff_leader_follower2",
    "speed_diff_leader_follower3", "speed_diff_leader_follower4",
    "predicted_spacing", "spacing_residual", "spacing_residual_abs",
    "cacc_accel_residual", "expected_accel_from_cacc",
    "bsm_jerk", "bsm_jerk_abs", "platoon_jerk",
    "minimum_spacing", "maximum_spacing", "spacing_range",
    "cross_receiver_speed_std",
    "predicted_speed_bsm", "speed_prediction_residual",
]

# SECTION D: temporal rolling features
# These are dynamically named (e.g. bsm_speed_roll_mean) to collect
# them at runtime rather than hardcoding every combination here
TEMPORAL_SUFFIXES = ["_roll_mean", "_roll_std", "_roll_rms"]

# combined static list (temporal features added dynamically in get_feature_cols)
ALL_STATIC_FEATURES = PLATOON_FEATURES + BSM_FEATURES + CONSISTENCY_FEATURES

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    '''
    returns the list of columns in df that should be normalized

    strategy:
        1. start with the static feature lists above
        2. add any temporal rolling features (detected by suffix)
        3. add any remaining numeric columns not in NEVER_NORMALIZE
        4. filter to only columns actually present in df
        5. verify no identifier/label columns
    
    this is called with the training dataframe so the feature list is determined from real data
    '''

    # start with static lists
    candidates = set(ALL_STATIC_FEATURES)

    # add temporal features dynamically (bsm_speed_roll_mean etc.)
    for col in df.columns:
        if any(col.endswith(suffix) for suffix in TEMPORAL_SUFFIXES):
            candidates.add(col)
        # also catch bsm_speed_trend and similar
        if col.endswith("_trend"):
            candidates.add(col)

    # add any numeric column not explicitly protected
    # this is a safety net for new features added to features.py
    # that aren't yet in the static lists above
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col not in NEVER_NORMALIZE:
            candidates.add(col)
    
    # filter to columns present in df, sorted for reproducibility
    feature_cols = sorted([c for c in candidates if c in df.columns])

    # final guard: remove anything that slipped through NEVER_NORMALIZE
    feature_cols = [c for c in feature_cols if c not in NEVER_NORMALIZE]

    log(f"Normalization: {len(feature_cols)} feature columns identified")
    return feature_cols

# scaler construction
def make_scaler(method: str):
    '''
    returns a fresh unfitted sklearn scaler

    zscore (StandardScaler)
      - outputs ~N(0,1) which keeps LSTM gate activations in linear regime
      - sensitive to outliers (attack injections are outliers by design —
        this is fine, the model needs to see them as extreme values)

    minmax (MinMaxScaler): use for bounded physical signals
      - outputs [0,1], preserves relative magnitude
      - attack outliers can push test values outside [0,1] if magnitudes
        exceed training range
    '''
    method = method.lower().strip()
    if method in ("zscore", "z-score", "standard"):
        return StandardScaler()
    if method in ("minmax", "min-max", "minmax"):
        return MinMaxScaler()
    raise ValueError(
        f"Unknown normalization method: '{method}'. "
        f"Choose 'zscore' or 'minmax'."
    )

# fit / transform / inverse
def fit_scaler(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    method: str = "zscore",
):
    '''
    fits a scaler on training data only

    ***parameters***
    train_df:       the training split DataFrame (never val or test)
    feature_cols:   columns to fit on (from get_feature_cols)
    method:         'zcore' or 'minmax'

    ***returns***
    fitted scaler object
    '''

    # validate no NaN/inf in training features before fitting
    train_vals = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    n_nan = train_vals.isna().sum().sum()
    if n_nan > 0:
        log(f"[WARN] {n_nan} NaN/inf values in training features before fit — "
            f"filling with column median for scaler fitting only")
        train_vals = train_vals.fillna(train_vals.median())

    scaler = make_scaler(method)
    scaler.fit(train_vals.values)

    # log fitted statistics for audit trail
    if hasattr(scaler, "mean_"):
        log(f"Scaler fitted: method={method}, "
            f"mean range=[{scaler.mean_.min():.3f}, {scaler.mean_.max():.3f}], "
            f"std range=[{scaler.scale_.min():.3f}, {scaler.scale_.max():.3f}]")
    elif hasattr(scaler, "data_min_"):
        log(f"Scaler fitted: method={method}, "
            f"data_min range=[{scaler.data_min_.min():.3f}, {scaler.data_min_.max():.3f}], "
            f"data_max range=[{scaler.data_max_.max():.3f}]")

    return scaler

def transform(
    df: pd.DataFrame,
    scaler,
    feature_cols: list[str],
) -> pd.DataFrame:
    '''
    applies a fitted scaler to df[feature_cols]
    all other columns unchanged
    handles NaN/inf by filling with 0 after scaling (not before, to avoid
        contaminating the scaler statistics with imputed values)
    '''
    df = df.copy()
    vals = df[feature_cols].replace([np.inf, -np.inf], np.nan).values
    scaled = scaler.transform(vals)
    # fill any NaN that survived (e.g. from a column with zero variance
    # in a small val/test group) with 0 (= mean in zscore space)
    scaled = np.where(np.isnan(scaled), 0.0, scaled)
    df[feature_cols] = scaled
    return df

def inverse_transform(
    df: pd.DataFrame,
    scaler,
    feature_cols: list[str],
) -> pd.DataFrame:
    '''reverses scaling — used for interpreting model outputs in physical units'''
    df = df.copy()
    df[feature_cols] = scaler.inverse_transform(df[feature_cols].values)
    return df

# persistence
def save_scaler(scaler, path: str | Path):
    '''saves fitted scaler to disk as a pickle file'''
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(scaler, f)
    log(f"Saved scaler -> {path}")


def load_scaler(path: str | Path):
    '''loads a previously fitted scaler from disk'''
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found at {path}")
    with path.open("rb") as f:
        scaler = pickle.load(f)
    log(f"Loaded scaler <- {path}")
    return scaler

def save_feature_cols(feature_cols: list[str], path: str | Path):
    '''
    saves the feature column list alongside the scaler
    critical for inference: the model must receive features in exactly
        the same order and set as during training
    '''
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(feature_cols, f, indent=2)
    log(f"Saved feature column list ({len(feature_cols)} cols) -> {path}")

def load_feature_cols(path: str | Path) -> list[str]:
    '''loads the saved feature column list for inference'''
    with open(path) as f:
        return json.load(f)
    
# orchestration: fit on train and apply to all splits
def normalize(
    engineered: pd.DataFrame | None = None,
    train_global_run_ids: set | None = None,
    method: str | None = None,
) -> pd.DataFrame:
    '''
    fits a scaler on the training rows of engineered DataFrame,
    applies it to all rows, and saves the scaler + feature list

    ***parameters***
    engineered          : feature-engineered DataFrame (all splits combined)
    train_global_run_ids: set of GlobalRunIDs belonging to the train split
                          (used to fit scaler on train rows only)
                          if None, fits on the entire DataFrame --> only
                          acceptable for exploratory runs, not final pipeline
    method              : 'zscore' or 'minmax' (defaults to CONFIG.norm_method)

   ***returns***
    normalized DataFrame with same shape as input
    '''
    from helpers.io import load_csv, save_dataframe

    if engineered is None:
        engineered = load_csv(CONFIG.engineered_dir / CONFIG.engineered_file)

    method = method or CONFIG.norm_method
    feature_cols = get_feature_cols(engineered)

    # fit on train split only
    if train_global_run_ids is not None:
        train_mask = engineered["GlobalRunID"].isin(train_global_run_ids)
        train_df   = engineered[train_mask]
        log(f"Fitting scaler on {train_mask.sum():,} training rows "
            f"({train_df['GlobalRunID'].nunique()} runs)")
    else:
        log("[WARN] train_global_run_ids not provided — fitting scaler on ALL data. "
            "This is only acceptable for exploratory use, not final pipeline.")
        train_df = engineered

    scaler = fit_scaler(train_df, feature_cols, method=method)

    # apply to all rows (train + val + test)
    normalized = transform(engineered, scaler, feature_cols)

    # save scaler and feature list for inference
    ensure_dir(CONFIG.normalized_dir)
    save_scaler(scaler,        CONFIG.normalized_dir / "scaler.pkl")
    save_feature_cols(feature_cols, CONFIG.normalized_dir / "feature_cols.json")

    out_path = CONFIG.normalized_dir / CONFIG.normalized_file
    save_dataframe(normalized, out_path)
    log(f"Saved normalized data -> {out_path} "
        f"({len(normalized):,} rows, {len(feature_cols)} normalized features)")

    return normalized

if __name__ == "__main__":
    normalize()