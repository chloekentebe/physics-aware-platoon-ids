'''Purpose: transform raw sychronized signals into physics-aware features.
Input: sychronized DataFrame (platoon + BSM signals on shared timeline)
Output: feature-engineered DataFrame ready for normalization and windowing

Internal structure:
    Section A - Raw platoon signals     (ground truth vehicle state)
    Section B - BSM communication       (what vehicles claim to be doing)
    Section C - Physics consistency     (disagreement between A and B)
    Section D - Derived temporal        (rolling statistics for BiLSTM context)

The IDS signal lives primarily in Section C: an honest vehicle's BSM should
agree with the platoon ground truth within sensor noise. Attacks create systematic
deviations that show up as non-zero consistency errors.
'''

import numpy as np
import pandas as pd

from config import CONFIG
from helpers.io import load_csv, save_dataframe
from helpers.physics import(
    absolute_error, relative_speed, relative_acceleration, closing_rate,
    time_headway, spacing_error, speed_error, predicted_spacing, predicted_speed,
    residual, jerk, rolling_rms, safe_divide
)
from helpers.utils import ensure_dir, log

# column name maps = single source of truth for all feature name convention

# ground truth platoon speeds per vehicle (from platoon CSV after sync)
PLATOON_SPEED_COLS = {
    "Leader": "platoon_speed_leader",
    "Follower1": "platoon_speed_follower1",
    "Follower2": "platoon_speed_follower2",
    "Follower3": "platoon_speed_follower3",
    "Follower4": "platoon_speed_follower4"
}

# ground truth platoon accelerations per vehicle
PLATOON_ACCEL_COLS = {
    "Leader":    "platoon_accel_leader",
    "Follower1": "platoon_accel_follower1",
    "Follower2": "platoon_accel_follower2",
    "Follower3": "platoon_accel_follower3",
    "Follower4": "platoon_accel_follower4",
}

# intervehicle spacing columns (from platoon CSV)
SPACING_COLS = {
    "Follower1": "spacing_l_f1",
    "Follower2": "spacing_f1_f2",
    "Follower3": "spacing_f2_f3",
    "Follower4": "spacing_f3_f4",
}

# BSM-reported fields (from BSM CSV after sync, J2735-decoded to physical units)
BSM_SPEED_COL   = "bsm_speed"          # m/s
BSM_ACCEL_COL   = "bsm_long_accel"     # m/s²
BSM_LAT_COL     = "bsm_latitude"       # degrees
BSM_LON_COL     = "bsm_longitude"      # degrees
BSM_HEADING_COL = "bsm_heading"        # degrees
BSM_MSGCNT_COL  = "bsm_msg_count"

# grouping keys used when computer per-channel rolling/diff statistics
# a "channel" is one (run, receiver, sender) communication link
GROUP_COLS_BSM     = ["GlobalRunID", "ReceiverID", "SenderID"]
GROUP_COLS_PLATOON = ["GlobalRunID"]

# Earth radius constant for position consistency (flat-earth approximation)
EARTH_M_PER_DEG_LAT = 111_320.0

# SECTION A: raw platoon features
# keep ground-truth vehicle state as is --> these are the reference that
# SECTION C comparisons are made against

def add_section_a_raw_platoon(df: pd.DataFrame) -> pd.DataFrame:
    '''
    rename and validates raw platoon signals so downstream code uses consistent
    consistent column names regardless of how the MATLAB CSV named them

    also derives platoon-level accelerations from velocity
    '''
    df = df.copy()

    # velocity rename map: MATLAB name -> standardized name
    matlab_speed_rename = {
        "leaderVal":    "platoon_speed_leader",
        "follower1Val": "platoon_speed_follower1",
        "follower2Val": "platoon_speed_follower2",
        "follower3Val": "platoon_speed_follower3",
        "follower4Val": "platoon_speed_follower4",
    }
    for src, dst in matlab_speed_rename.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # spacing rename map
    matlab_spacing_rename = {
        "spacing1Val": "spacing_l_f1",
        "spacing2Val": "spacing_f1_f2",
        "spacing3Val": "spacing_f2_f3",
        "spacing4Val": "spacing_f3_f4",
    }
    for src, dst in matlab_spacing_rename.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # derived platoon acceleration via finite difference on velocity
    dt = CONFIG.platoon_time_step  # 0.2s — platoon signal native sample rate
    for vehicle, speed_col in PLATOON_SPEED_COLS.items():
        accel_col = PLATOON_ACCEL_COLS[vehicle]
        if speed_col in df.columns and accel_col not in df.columns:
            df[accel_col] = (
                df.groupby(GROUP_COLS_PLATOON)[speed_col]
                .diff()
                .fillna(0)
                .div(dt)
            )

    log("Section A: raw platoon features ready")
    return df

# SECTION B: BSM communication features
# keep all BSM-reported values and derive simple communication health metrics

def add_section_b_communication(df: pd.DataFrame) -> pd.DataFrame:
    '''
    adds BSM-level features:
    - J2735-decoded physical values (should already be decoded in sync step,
      but it's being guarded in case they're still raw integers)
    - message count delta (should be exactly 1 per step; deviations flag replay/freeze
      attacks)
    - BSM-derived acceleration (finite difference of reported speed)
    - heading change rate
    '''
    df = df.copy()

    # guard: decode J2735 integers to physical units
    # (synchronization.py should handle this, but deferensive here)
    if BSM_SPEED_COL in df.columns:
        # if values look like raw J2735 (range ~0-8191 for speed), decode
        if df[BSM_SPEED_COL].max() > 200:
            df[BSM_SPEED_COL] = df[BSM_SPEED_COL] * 0.02      # -> m/s
    if BSM_ACCEL_COL in df.columns:
        if df[BSM_ACCEL_COL].abs().max() > 50:
            df[BSM_ACCEL_COL] = df[BSM_ACCEL_COL] * 0.01     # -> m/s²
    if BSM_LAT_COL in df.columns:
        if df[BSM_LAT_COL].abs().max() > 200:
            df[BSM_LAT_COL] = df[BSM_LAT_COL] * 1e-7         # -> degrees
    if BSM_LON_COL in df.columns:
        if df[BSM_LON_COL].abs().max() > 500:
            df[BSM_LON_COL] = df[BSM_LON_COL] * 1e-7         # -> degrees
    if BSM_HEADING_COL in df.columns:
        if df[BSM_HEADING_COL].max() > 360:
            df[BSM_HEADING_COL] = df[BSM_HEADING_COL] * 0.0125  # -> degrees
    
    # message count delta: should be exactly +1 each step (wraps at 128)
    # MsgCntFDI attack freezes/replays the counter -> delta != 1
    if BSM_MSGCNT_COL in df.columns:
        raw_delta = (
            df.groupby(GROUP_COLS_BSM)[BSM_MSGCNT_COL]
            .diff()
            .fillna(1)
        )
        # handle wrap-around: 127 -> 0 is a valid +1 step, not a -127 drop
        df["bsm_msg_count_delta"] = np.where(
            raw_delta < -64, raw_delta + 128, raw_delta
        )
        # expected delta is always 1; deviation from 1 is the anomaly signal
        df["bsm_msg_count_deviation"] = df["bsm_msg_count_delta"] - 1.0

    # BSM-derived acceleration from reported speed (finite difference)
    # distinct from bsm_long_accel: that's what the vehicle CLAIMS its
    # acceleration is; this is what the speed IMPLIES the acceleration is
    if BSM_SPEED_COL in df.columns:
        dt = CONFIG.bsm_time_step  # 0.02s
        df["bsm_derived_accel"] = (
            df.groupby(GROUP_COLS_BSM)[BSM_SPEED_COL]
            .diff()
            .fillna(0)
            .div(dt)
        )
    
    # heading change rate (yaw rate proxy from BSM heading)
    if BSM_HEADING_COL in df.columns:
        dt = CONFIG.bsm_time_step
        df["bsm_heading_rate"] = (
            df.groupby(GROUP_COLS_BSM)[BSM_HEADING_COL]
            .diff()
            .fillna(0)
            .div(dt)
        )

    log("Section B: BSM communication features ready")
    return df

# SECTION C: physics consistency features
# these are the core IDS signals - each one measures a physical invariant
# that must hold for an honest vehicle but is violated by specific attacks
def _assign_sender_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    '''
    helper: for each BSM row, look up the GROUND TRUTH speed and acceleration
    of the SENDER vehicle from the platoon columns

    purpose: the BSM table is in the long format (one row per receiver-sender per timestep),
    but platoon ground truth has one column per vehicle
    
    the right platoon ground truth needs to be matched to each BSM row based on which
    vehicle is the sender
    '''
    df["sender_platoon_speed"] = np.nan
    df["sender_platoon_accel"] = np.nan

    for vehicle, speed_col in PLATOON_SPEED_COLS.items():
        accel_col = PLATOON_ACCEL_COLS[vehicle]
        if speed_col in df.columns and "SenderID" in df.columns:
            mask = df["SenderID"].astype(str) == vehicle
            df.loc[mask, "sender_platoon_speed"] = df.loc[mask, speed_col]
        if accel_col in df.columns and "SenderID" in df.columns:
            mask = df["SenderID"].astype(str) == vehicle
            df.loc[mask, "sender_platoon_accel"] = df.loc[mask, accel_col]

    return df

def _assign_receiver_spacing(df: pd.DataFrame) -> pd.DataFrame:
    '''
    helper: for each BSM row, look up at the ground-truth intervehicle spacing
    for the RECEIVER vehicle (spacing between receiver and the vehicle ahead)
    '''
    df["receiver_spacing"] = np.nan
    for vehicle, spacing_col in SPACING_COLS.items():
        if spacing_col in df.columns and "ReceiverID" in df.columns:
            mask = df["ReceiverID"].astype(str) == vehicle
            df.loc[mask, "receiver_spacing"] = df.loc[mask, spacing_col]
    return df

def add_section_c_physics_consistency(df: pd.DataFrame) -> pd.DataFrame:
    '''
    computes all physics consistency features using helper from physics.py

    every feature here answers the question:
        'Does the BSM-reported value agree with what is physically excpected?'
    
    non-zero values flag potential attacks or sensor faults

    the BiLSTM learns to distinguish systematic attack patterns from random sensor noise
    by observing these features across the temporal window
    '''

    # C1. SPEED CONSISTENCY
    # BSM-reported speed vs ground-truth platoon speed of the same sender
    # SpeedFDI attack: df["speed_consistency_error"] deviates from ~0
    if BSM_SPEED_COL in df.columns and "sender_platoon_speed" in df.columns:
        df["speed_consistency_error"] = speed_error(
            df[BSM_SPEED_COL], df["sender_platoon_speed"]
        )
        df["speed_consistency_abs_error"] = absolute_error(
            df[BSM_SPEED_COL], df["sender_platoon_speed"]
        )
        # Relative error: error as fraction of true speed
        df["speed_consistency_rel_error"] = safe_divide(
            df["speed_consistency_abs_error"], df["sender_platoon_speed"].abs()
        )

    # C2. ACCELERATION CONSISTENCY
    # BSM-reported LongAcc vs ground-truth platoon acceleration
    # AccFDI attack: df["accel_consistency_error"] deviates from ~0
    # also checks internal BSM consistency: LongAcc should match dSpeed/dt
    if BSM_ACCEL_COL in df.columns and "sender_platoon_accel" in df.columns:
        df["accel_consistency_error"] = absolute_error(
            df[BSM_ACCEL_COL], df["sender_platoon_accel"]
        )

    if BSM_ACCEL_COL in df.columns and "bsm_derived_accel" in df.columns:
        # internal BSM inconsistency: reported accel vs speed-implied accel
        # (catches attacks that modify LongAcc but forget to also adjust Speed)
        df["bsm_internal_accel_error"] = absolute_error(
            df[BSM_ACCEL_COL], df["bsm_derived_accel"]
        )
    
    # C3. POSITION CONSISTENCY
    # how far does the BSM-claimed position deviate from the expected
    # position based on the last known position + claimed speed?
    # PosFDI attack: position jumps discontinuously.
    if {BSM_LAT_COL, BSM_LON_COL, BSM_SPEED_COL}.issubset(df.columns):
        dt = CONFIG.bsm_time_step

        # convert lat/lon displacement to metres (flat-earth approximation)
        lat_rad = np.radians(df[BSM_LAT_COL])
        dlat_m = (
            df.groupby(GROUP_COLS_BSM)[BSM_LAT_COL]
            .diff()
            .fillna(0)
            * EARTH_M_PER_DEG_LAT
        )
        dlon_m = (
            df.groupby(GROUP_COLS_BSM)[BSM_LON_COL]
            .diff()
            .fillna(0)
            * EARTH_M_PER_DEG_LAT
            * np.cos(lat_rad)
        )
        df["bsm_position_displacement"] = np.sqrt(dlat_m**2 + dlon_m**2)

        # expected displacement from reported speed
        df["bsm_expected_displacement"] = df[BSM_SPEED_COL] * dt

        # position-velocity consistency: displacement should match speed
        df["position_velocity_consistency"] = absolute_error(
            df["bsm_position_displacement"],
            df["bsm_expected_displacement"]
        )
    
    # C4. TIME HEADWAY AND SPACING
    # time headway = gap / speed
    # standard platooning target is ~1-2s
    # deviations indicate spacing anomalies consistent with false speed reports
    if "receiver_spacing" in df.columns and BSM_SPEED_COL in df.columns:
        df["time_headway"] = time_headway(
            df["receiver_spacing"], df[BSM_SPEED_COL]
        )
        # 2-second rule desired spacing: desired_gap = 2.0 * v
        df["spacing_error_2s"] = spacing_error(
            df["receiver_spacing"], 2.0 * df[BSM_SPEED_COL]
        )
        df["spacing_error_abs"] = absolute_error(
            df["receiver_spacing"], 2.0 * df[BSM_SPEED_COL]
        )
    
    # C5. RELATIVE SPEED AND CLOSING RATE
    # between consecutive vehicles
    # non-zero closing rate means the platoon
    # is not maintaining constant spacing — expected transiently, but
    # sustained closing rate under attack reveals false speed claims
    speed_pairs = [
        ("Leader",    "Follower1", "platoon_speed_leader",    "platoon_speed_follower1"),
        ("Follower1", "Follower2", "platoon_speed_follower1", "platoon_speed_follower2"),
        ("Follower2", "Follower3", "platoon_speed_follower2", "platoon_speed_follower3"),
        ("Follower3", "Follower4", "platoon_speed_follower3", "platoon_speed_follower4"),
    ]
    for front, rear, front_col, rear_col in speed_pairs:
        tag = f"{front.lower()}_{rear.lower()}"
        if {front_col, rear_col}.issubset(df.columns):
            df[f"relative_speed_{tag}"] = relative_speed(
                df[front_col], df[rear_col]
            )
            df[f"closing_rate_{tag}"] = closing_rate(
                df[front_col], df[rear_col]
            )
            df[f"relative_accel_{tag}"] = relative_acceleration(
                df[PLATOON_ACCEL_COLS[front]], df[PLATOON_ACCEL_COLS[rear]]
            ) if {PLATOON_ACCEL_COLS[front], PLATOON_ACCEL_COLS[rear]}.issubset(df.columns) else 0

    # C6. LEADER-FOLLOWER SPEED DIFFERENCES
    # each follower's speed vs leader
    # under SpeedFDI on the leader,
    # the followers' CACC reacts to a false leader speed — this residual
    # captures the induced tracking error across the platoon
    if "platoon_speed_leader" in df.columns:
        for vehicle, speed_col in PLATOON_SPEED_COLS.items():
            if vehicle == "Leader" or speed_col not in df.columns:
                continue
            df[f"speed_diff_leader_{vehicle.lower()}"] = speed_error(
                df["platoon_speed_leader"], df[speed_col]
            )
    
    # C7. PREDICTED SPACING AND RESIDUAL
    # given current spacing and relative speed, predict spacing one step ahead
    # residual between prediction and observed spacing reveals dynamic anomalies
    if {"receiver_spacing", "relative_speed_leader_follower1"}.issubset(df.columns):
        prev_spacing = (
            df.groupby(GROUP_COLS_BSM)["receiver_spacing"]
            .shift(1)
            .fillna(df["receiver_spacing"])
        )
        rel_spd = df.get(
            "relative_speed_leader_follower1",
            pd.Series(0.0, index=df.index)
        )
        df["predicted_spacing"] = predicted_spacing(
            prev_spacing, rel_spd, CONFIG.bsm_time_step
        )
        df["spacing_residual"] = residual(
            df["receiver_spacing"], df["predicted_spacing"]
        )
        df["spacing_residual_abs"] = df["spacing_residual"].abs()
    
    # C8. EXPECTED ACCELERATION FROM CACC PHYSICS
    # the CACC controller formula: a_cmd = 0.8*a_lead + 0.2*a_prev - 8*(v_ego - v_lead) - k_p*e
    # compute the EXPECTED acceleration from ground truth and compare to BSM-reported
    # significant deviation reveals false acceleration claims.
    if {
        "platoon_accel_leader", "platoon_speed_leader",
        BSM_ACCEL_COL, BSM_SPEED_COL
    }.issubset(df.columns) and "sender_platoon_speed" in df.columns:
        # simplified expected: if sender is honest, BSM accel should track
        # platoon accel within sensor noise
        df["expected_accel_from_cacc"] = (
            0.8 * df["platoon_accel_leader"]
        )
        df["cacc_accel_residual"] = residual(
            df[BSM_ACCEL_COL], df["expected_accel_from_cacc"]
        )

    # C9. JERK (rate of change of acceleration)
    # high jerk indicates sudden unrealistic acceleration changes.
    # useful for detecting abrupt FDI injections that create unphysical steps.
    if BSM_ACCEL_COL in df.columns:
        df["bsm_jerk"] = df.groupby(GROUP_COLS_BSM)[BSM_ACCEL_COL].transform(
            lambda s: pd.Series(
                jerk(s.to_numpy(), CONFIG.bsm_time_step), index=s.index
            )
        )
        df["bsm_jerk_abs"] = df["bsm_jerk"].abs()

    if "sender_platoon_accel" in df.columns:
        df["platoon_jerk"] = df.groupby(GROUP_COLS_PLATOON)["platoon_accel_leader"].transform(
            lambda s: pd.Series(
                jerk(s.to_numpy(), CONFIG.platoon_time_step), index=s.index
            )
        ) if "platoon_accel_leader" in df.columns else 0.0

    # C10. MONOTONIC SPACING CHECK
    # in a well-behaved platoon, spacings should decrease monotonically
    # from leader to last follower (vehicles tighten up from front to rear)
    # violation indicates disordered platoon state.
    spacing_col_list = [c for c in SPACING_COLS.values() if c in df.columns]
    if len(spacing_col_list) >= 2:
        df["minimum_spacing"] = df[spacing_col_list].min(axis=1)
        df["maximum_spacing"] = df[spacing_col_list].max(axis=1)
        df["spacing_range"]   = df["maximum_spacing"] - df["minimum_spacing"]
        # 1 if spacings are monotonically non-increasing front-to-back, else 0
        diffs = df[spacing_col_list].diff(axis=1).iloc[:, 1:]
        df["spacing_monotonic"] = (diffs <= 0).all(axis=1).astype(int)
    
    # C11. CROSS-VEHICLE BSM SPEED CONSISTENCY
    # multiple receivers see the same sender's BSM
    # if all receivers get the same Speed value from the same sender,
    # std dev across receivers at the same timestep should be ~0
    # non-zero std indicates inconsistency
    # (relevant for channel-induced corruption or multi-receiver injection).
    if {BSM_SPEED_COL, "SenderID", "time"}.issubset(df.columns):
        cross_std = (
            df.groupby(["GlobalRunID", "time", "SenderID"])[BSM_SPEED_COL]
            .transform("std")
            .fillna(0)
        )
        df["cross_receiver_speed_std"] = cross_std

    log("Section C: physics consistency features ready")
    return df

# SECTION D: derived temporal features
# rolling statistics give the BiLSTM short-term context about trends,
# variance, and signal stability - features the raw instantaneous values
# which alone they cannot express
def add_section_d_temporal(df: pd.DataFrame) -> pd.DataFrame:
    '''
    computes rolling statistics over a configurable window (CONFIG.rolling_window)

    applied to the most informative consistency features from Section C, plus raw BSM
    speed and acceleration

    section A/B are not all rolled; the BiLSTM temporal window already captures their history

    the features where local trend or variance is the signal is rolled
    '''
    df = df.copy()
    w = CONFIG.rolling_window # e.g. 5 timsteps = 0.1s at 0.02s BSM rate

    # columns to apply temporal statistics to
    temporal_targets = [
        BSM_SPEED_COL,
        BSM_ACCEL_COL,
        "speed_consistency_error",
        "speed_consistency_abs_error",
        "accel_consistency_error",
        "position_velocity_consistency",
        "time_headway",
        "spacing_error_2s",
        "bsm_jerk",
        "spacing_residual",
        "bsm_msg_count_deviation",
        "cross_receiver_speed_std",
    ]

    # only process columns that actually exist after Section C
    targets = [c for c in temporal_targets if c in df.columns]

    for col in targets:
        grouped = df.groupby(GROUP_COLS_BSM)[col]

        # rolling mean: tracks the local level (smoothed signal)
        df[f"{col}_roll_mean"] = grouped.transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        # rolling std: tracks local variability (FDI attacks often reduce
        # variance by injecting a constant bias, or increase it abruptly)
        df[f"{col}_roll_std"] = grouped.transform(
            lambda s: s.rolling(w, min_periods=1).std().fillna(0)
        )
        # rolling RMS: energy of the signal — useful for zero-mean signals
        # like jerk and consistency errors where mean is uninformative
        df[f"{col}_roll_rms"] = grouped.transform(
            lambda s: rolling_rms(s, w)
        )
    
    # short-term acceleration trend: linear regression slope over window
    # a rising slope under attack reveals systematic velocity manipulation
    if BSM_SPEED_COL in df.columns:
        def _slope(s):
            arr = s.to_numpy()
            n   = len(arr)
            slopes = np.zeros(n)
            for i in range(n):
                start = max(0, i - w + 1)
                chunk = arr[start:i + 1]
                if len(chunk) >= 2:
                    slopes[i] = np.polyfit(range(len(chunk)), chunk, 1)[0]
            return pd.Series(slopes, index=s.index)

        df["bsm_speed_trend"] = df.groupby(GROUP_COLS_BSM)[BSM_SPEED_COL].transform(_slope)

    log(f"Section D: temporal features ready "
        f"(rolling window={w}, {len(targets)} base columns x 3 statistics)")
    return df

# orchestration
def engineer_features(synchronized: pd.DataFrame | None = None) -> pd.DataFrame:
    '''
    runs all four sections in order on the sychronized DataFrame
    if synchronized is None, loads from CONFIG paths

    returns the feature-engineered DataFrame and saves it to disk
    '''
    if synchronized is None:
        synchronized = load_csv(CONFIG.sychronized_dir / CONFIG.sychronized_file)
    
    log(f"Starting feature engineering: {len(synchronized):,} rows, "
        f"{synchronized.columns.tolist().__len__()} input columns")

    df = add_section_a_raw_platoon(synchronized)
    df = add_section_b_communication(df)
    df = add_section_c_physics_consistency(df)
    df = add_section_d_temporal(df)

    # clean up: replace inf/-inf with NaN then fill with 0
    # inf can appear from safe_divide when denominator is exactly 0

    n_original = synchronized.shape[1]
    n_new = df.shape[1] - n_original
    log(f"Feature engineering complete: {n_original} input -> "
        f"{df.shape[1]} total columns (+{n_new} engineered)")

    ensure_dir(CONFIG.engineered_dir)
    out_path = CONFIG.engineered_dir / CONFIG.engineered_file
    save_dataframe(df, out_path)

    log(f"saved engineered features -> {out_path}")
    return df

if __name__ == "__main__":
    engineer_features()