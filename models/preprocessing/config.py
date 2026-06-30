'''Purpose: stores every configurable parameter for the preprocessing pipeline,
'''

from dataclasses import dataclass
from pathlib import Path

PREPROCESSING_DIR = Path(__file__).resolve().parent
REPO_ROOT = PREPROCESSING_DIR.parents[1]

@dataclass(frozen=True)
class PreprocessingConfig:
    # RAW DATA DIRECTORIES
    normal_dir: Path = REPO_ROOT / "simulation/matlab/normal/normal_dataset"
    v2i_dir: Path = REPO_ROOT / "simulation/matlab/v2i/attack_dataset"
    v2v_dir: Path = REPO_ROOT / "simulation/matlab/v2v/attack_dataset"

    # PROCESSED OUTPUT DIRECTORIES
    processed_dir: Path = PREPROCESSING_DIR / "processed"
    merged_dir: Path = processed_dir / "merged"
    synchronized_dir: Path = processed_dir / "synchronized"
    engineered_dir: Path = processed_dir / "engineered"
    normalized_dir: Path = processed_dir / "normalized"
    windows_dir: Path = processed_dir / "windows"
    final_dir: Path = processed_dir / "final"

    # FILE NAMES
    merged_file: str = "merged.parquet"
    synchronized_file: str = "synchronized.parquet"
    engineered_file: str = "features.parquet"
    normalized_file: str = "normalized.parquet"
    windows_file: str = "windows.npz"

    # SAMPLING RATES
    bsm_time_step:       float = 0.02   
    platoon_time_step:   float = 0.2  
    sim_duration:        float = 60.0
    time_round_decimals: int   = 2

    # SLIDING WINDOW
    window_size: int = 250
    stride: int = 125

    # ROLLING STATISTICS WINDOW
    rolling_window: int = 5

    # NORMALIZATION
    norm_method: str = "zscore"

    # TRAIN/VAL/TEST SPLIT
    train_fraction: float = 0.70
    val_fraction:   float = 0.15
    test_fraction:  float = 0.15
    random_seed:    int = 42

    # CACC CONTROLLER CONSTANTS
    cacc_c1: float = 0.8 # leder acceleration weight
    cacc_k1: float = 8.0 # velocity error gain
    cacc_kp: float = 2.0 # proportional spacing gain

    # COLUMNS THAT ARE NEVER NORMALIZED/USED AS BILSTM FEATURES
    id_columns: tuple = (
        # run identity
        "GlobalRunID", "RunID", "loop_idx",
        "dataset", "source_file", "source_type",
        # Sscenario descriptors
        "Profile", "ScenarioType",
        # vehicle/communication identity
        "ReceiverID", "SenderID", "SenderSlot", "bsm_vehicle_id",
        # attack labels — all hierarchy levels
        "is_attack", "attack_vector", "attack_type", "attacker_id", "attack_active",
        "IsAttack", "AttackVector", "AttackType", "AttackerID", "AttackActive",
        "IsSenderAttacker",
        # attack metadata
        "AttackMag", "AttackStart", "AttackDuration",
        "TimingShift1", "TimingShift2", "TimingShift3",
        "AttackDelta1", "AttackDelta2", "AttackDelta3",
        # time — it's the index
        "time", "Time",
        # BSM protocol enum fields
        "bsm_sec_mark",
    )

CONFIG = PreprocessingConfig()