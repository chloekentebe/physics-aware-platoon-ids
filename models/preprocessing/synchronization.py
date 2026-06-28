'''Purpose: put every signal onto the same timeline.
    Platoon -> 0.2s seconds native rate
    BSM -> 0.02 seconds (snapped to clean grid)

Strategy:
    - BSM timestamps are the MASTER grid (finer resolution amd attack signals live here)
    - Platoon signals are interpolated onto the BSM grid (valid since velocity/spacing are
      smooth signals --> linear interpolation over 0.18s introduces negligible error)
    - Synchronized per (GlobalRunID, ReceiverID, SendorSlot) channel so no cross-channel
    contamination

Output: one row per (GlobalRunID, time, ReceiverID, SenderSlot) with both platoon ground truth
        and BSM communication signals present'''

import pandas as pd
import numpy as np

from config import CONFIG
from helpers.io import load_csv, save_dataframe
from helpers.utils import ensure_dir, log
from helpers.validation import require_columns, require_monotonic, report_quality
from helpers.interpolation import interpolate_to_grid
from helpers.labels import NORMAL_LABEL_DEFAULTS

# column rename maps - applied before any preprocessing so downstream code always
# uses standardized names regarelss of MATLAB CSV naming

PLATOON_RENAME = {
    "leaderVal": "platoon_speed_leader",
    "follower1Val": "platoon_speed_follower1",
    "follower2Val": "platoon_speed_follower2",
    "follower3Val": "platoon_speed_follower3",
    "follower4Val": "platoon_speed_follower4",
    "spacing1Val": "spacing_l_f1",
    "spacing2Val": "spacing_f1_f2",
    "spacing3Val": "spacing_f2_f3",
    "spacing4Val": "spacing_f3_f4",
}

BSM_RENAME = {
    "Time":              "time",
    "Speed":             "bsm_speed",
    "Heading":           "bsm_heading",
    "Lattitude":         "bsm_latitude",
    "Longitude":         "bsm_longitude",
    "Elevation":         "bsm_elevation",
    "AccelSet_LongAcc":  "bsm_long_accel",
    "AccelSet_LatAcc":   "bsm_lat_accel",
    "AccelSet_VertAcc":  "bsm_vert_accel",
    "AccelSet_YawRate":  "bsm_yaw_rate",
    "MsgCnt":            "bsm_msg_count",
    "Id":                "bsm_vehicle_id",
    "SecMark":           "bsm_sec_mark",
    "TrueSpeed_ms":      "true_speed_ms",     # ground truth logged in MATLAB
    "ReportedSpeed_ms":  "reported_speed_ms",
    "SpeedDeviation":    "speed_deviation_raw",
}

# columns that carry run identity and labels - preserved through all merges
LABEL_COLS = [
    "GlobalRunID", "RunID", "loop_idx", "dataset", "Profile", "ScenarioType",
    "is_attack", "attack_vector", "attack_type", "attacker_id", "attack_active",
    "IsSenderAttacker", "AttackMag", "AttackStart", "AttackDuration",
    "TimingShift1", "TimingShift2", "TimingShift3",
    "AttackDelta1", "AttackDelta2", "AttackDelta3",
    "source_file",
]

def _fill_missing_labels(df: pd.DataFrame) -> pd.DataFrame:
    '''
    injects default label values for any  attack-label column that is absent

    called on both platoon and BSM rows so the synchronized output always has a complete,
    consistent label schema regardless of whether the source CSV came from a normal or attack
    generation script

    now encode_hierarchy() in save_dataset.py sees a uniform string vocabularly and builds
    one consistent integer mapping across all files
    '''
    for col, default in NORMAL_LABEL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    return df

def _prepare_platoon(merged: pd.DataFrame) -> pd.DataFrame:
    '''
    extracts platoon rows, renames columns to standardized names, and ensures time is numeric

    keeps all labels columns present
    '''
    # rename MATLAB column names to standardized names
    platoon = merged[merged["source_type"] == "platoon"].copy()
    platoon = platoon.rename(columns=PLATOON_RENAME)

    # sandardize time column name (MATLAB exports lowercase 'time')
    if "time" not in platoon.columns and "Time" in platoon.columns:
        platoon = platoon.rename(columns={"Time": "time"})

    platoon["time"] = pd.to_numeric(platoon["time"], errors="coerce")
    platoon = platoon.dropna(subset=["time"])

    # round timestamps to avoid floating point drift (13.099999 -> 13.10)
    platoon["time"] = platoon["time"].round(CONFIG.time_round_decimals)

    platoon = _fill_missing_labels(platoon)
    return platoon


def _prepare_bsm(merged: pd.DataFrame) -> pd.DataFrame:
    '''
    extracts BSM rows, renames to standardized names, ensures time is numeric,
    and rounds sporadic timestamps to the nearest clean grid point
    '''
    bsm = merged[merged["source_type"] == "bsm"].copy()
    bsm = bsm.rename(columns=BSM_RENAME)

    if "time" not in bsm.columns and "Time" in bsm.columns:
        bsm = bsm.rename(columns={"Time": "time"})

    bsm["time"] = pd.to_numeric(bsm["time"], errors="coerce")
    bsm = bsm.dropna(subset=["time"])

    # snap sporadic BSM timestamps to the nearest 0.02s grid point
    # e.g. 13.099999 -> 13.10, 0.019998 -> 0.02
    bsm["time"] = (bsm["time"] / CONFIG.bsm_time_step).round() * CONFIG.bsm_time_step
    bsm["time"] = bsm["time"].round(CONFIG.time_round_decimals)


    bsm = _fill_missing_labels(bsm)
    return bsm

def _get_platoon_value_cols(platoon: pd.DataFrame) -> list[str]:
    '''returns the platoon signal columns that will be interpolated onto BSM grid'''
    candidates = (
        list(PLATOON_RENAME.values()) +  # renamed versions
        [c for c in platoon.columns if c.startswith("platoon_") or c.startswith("spacing_")]
    )
    return [c for c in dict.fromkeys(candidates) if c in platoon.columns]

def _synchronize_one_run(
        bsm_run: pd.DataFrame,
        platoon_run: pd.DataFrame,
        global_run_id: str,
) -> pd.DataFrame:
    '''
    synchronize one run:
        1. build the master BSM time grid from BSM timestamps
        2. interpolate platoon signals onto that grid
        3. join interpolated platoon onto BSM (left join on time)
        4. validate monotonicity

    returns the synchronized DataFrame for this run or None of inputs are invalid
    '''
    if bsm_run.empty or platoon_run.empty:
        log(f"[SKIP] {global_run_id}; empty BSM or platoon data")
        return None

    # master grid: sorted unique BSM timestamps for this run
    grid = np.sort(bsm_run["time"].dropna().unique())
    if len(grid) == 0:
        log(f"[SKIP] {global_run_id}: no valid BSM timestamps")
        return None
    
    # platoon value columns to interpolate
    platoon_val_cols = _get_platoon_value_cols(platoon_run)

    # platoon label cols available in this run (for carry-through after merge)
    platoon_label_cols = [c for c in LABEL_COLS if c in platoon_run.columns]
    platoon_keep = ["time"] + platoon_label_cols + platoon_val_cols

    platoon_slim = platoon_run[[c for c in platoon_keep if c in platoon_run.columns]].copy()

    # interpolate platoon signals onto BSM grid
    # no group_cols needed here — platoon_run is already one run
    aligned_platoon = interpolate_to_grid(
        df=platoon_slim,
        time_col="time",
        grid=grid,
        group_cols=[],
    )

    # left join: every BSM row gets interpolated platoon values
    # BSM is left so every BSM row is kept even if platoon interpolation
    # missed a timestamp (shouldn't happen after grid alignment)
    synchronized = bsm_run.merge(
        aligned_platoon,
        on="time",
        how="left",
        suffixes=("", "_platoon"),
    )

    # drop duplicate label columns that came from platoon side of merge
    # (they're already present from BSM side, identical values)
    dup_cols = [c for c in synchronized.columns if c.endswith("_platoon")]
    synchronized = synchronized.drop(columns=dup_cols, errors="ignore")

    # validate monotonicity per BSM channel
    channel_cols = [c for c in ["ReceiverID", "SenderSlot", "SenderID"]
                    if c in synchronized.columns]
    try:
        require_monotonic(synchronized, "time", channel_cols,
                          context=f"run {global_run_id}")
    except ValueError as e:
        log(f"[WARN] {e} — sorting to fix")
        synchronized = synchronized.sort_values(channel_cols + ["time"]).reset_index(drop=True)

    return synchronized



def synchronize(merged: pd.DataFrame | None = None) -> pd.DataFrame:
    '''
    main synchronization function
        groups by GlobalRunID, synchronizes each run independently, and concatenates
    
    uses GlobalRunID as the join key throughout since
        - normal congested: platoon RunID = loopIdx+60, BSM RunID = loopIdx
        - normal deceleration: platoon RunID = loopIdx+90, BSM RunID = loopIdx
        - Global RunID encodes both table and profile so these never collid
    '''
    if merged is None:
        merged = load_csv(CONFIG.merged_dir / CONFIG.merged_file)
    
    require_columns(merged, ["GlobalRunID", "source_type"], "merged data for sync")

    platoon_all = _prepare_platoon(merged)
    bsm_all     = _prepare_bsm(merged)

    # validate both tables' existence before starting the loop
    platoon_run_ids = set(platoon_all["GlobalRunID"].unique())
    bsm_run_ids     = set(bsm_all["GlobalRunID"].unique())
    common          = platoon_run_ids & bsm_run_ids
    orphan_platoon  = platoon_run_ids - bsm_run_ids
    orphan_bsm      = bsm_run_ids - platoon_run_ids

    if orphan_platoon:
        log(f"[WARN] {len(orphan_platoon)} runs have platoon but no BSM — skipping")
    if orphan_bsm:
        log(f"[WARN] {len(orphan_bsm)} runs have BSM but no platoon — skipping")
    log(f"Synchronizing {len(common)} complete runs...")

    # pre-group once (O(n) total) rather than filtering inside loop (O(n*k))
    platoon_groups = dict(tuple(platoon_all.groupby("GlobalRunID")))
    bsm_groups     = dict(tuple(bsm_all.groupby("GlobalRunID")))

    synchronized_runs = []
    for i, run_id in enumerate(sorted(common)):
        result = _synchronize_one_run(
            bsm_run=bsm_groups[run_id],
            platoon_run=platoon_groups[run_id],
            global_run_id=run_id,
        )
        if result is not None:
            synchronized_runs.append(result)

        if (i + 1) % 50 == 0:
            log(f"  Synchronized {i + 1}/{len(common)} runs...")

    if not synchronized_runs:
        raise RuntimeError("No runs were successfully synchronized — check your data paths")

    synchronized = pd.concat(synchronized_runs, ignore_index=True, sort=False)

    # final sort: meaningful order for windowing downstream
    sort_cols = [c for c in ["GlobalRunID", "ReceiverID", "SenderSlot", "time"]
                 if c in synchronized.columns]
    synchronized = synchronized.sort_values(sort_cols).reset_index(drop=True)

    quality = report_quality(synchronized)
    log(f"Synchronization complete: {quality['rows']:,} rows, "
        f"{quality['columns']} columns, "
        f"{quality['missing_values']:,} missing values")

    ensure_dir(CONFIG.synchronized_dir)
    out_path = CONFIG.synchronized_dir / CONFIG.synchronized_file
    save_dataframe(synchronized, out_path)
    log(f"Saved synchronized data -> {out_path}")

    return synchronized


if __name__== "__main__":
    synchronize()