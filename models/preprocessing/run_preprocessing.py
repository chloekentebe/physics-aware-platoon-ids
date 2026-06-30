'''Purpose: coordinates the full preprocessing pipeline
    Order of piepline:
    1. merge            - load + label all CSVs
    2. synchronize      - align platoon (0.2s) and BSM (0.02s) onto shared grid
    3. engineer         - compute physics consistency features
    4. determine split  - identify train GlobalRunIDs before fitting scaler
    5. normalize        - fit scale on train only and apply to all
    6. windows          - slice into sliding windows
    7. save_data        - encode labels, split, write .npz files
'''
from config import CONFIG
from helpers.utils import ensure_dirs, timer, set_seed, log

from merge import merge
from synchronization import synchronize
from features import engineer_features
from normalization import normalize, get_feature_cols
from windows import create_temporal_windows
from save_dataset import save_dataset, stratified_run_split

def _get_train_run_ids(engineered) -> set:
    '''
    determines the train GlobalRunIDs via stratified split BEFORE normalization
        so the scaler is fit only on training data
        
    running the split here (on the engineered DataFrame before windowing) provides the
        required run-level IDs — the same IDs are reused in save_dataset to bucket windows
    '''
    run_meta = (
        engineered[["GlobalRunID", "Profile", "attack_type"]]
        .drop_duplicates("GlobalRunID")
        .set_index("GlobalRunID")
        .to_dict("index")
    )
    splits = stratified_run_split(run_meta, seed=CONFIG.random_seed)
    log(f"Train runs: {len(splits['train'])} | "
        f"Val runs: {len(splits['val'])} | "
        f"Test runs: {len(splits['test'])}")
    return splits["train"]

def run_preprocessing():
    set_seed(CONFIG.random_seed)

    ensure_dirs([
        CONFIG.merged_dir,
        CONFIG.synchronized_dir,
        CONFIG.engineered_dir,
        CONFIG.normalized_dir,
        CONFIG.windows_dir,
        CONFIG.final_dir
    ])

    with timer("merge"):
        merged = merge()
    
    with timer("synchronize"):
        synchronized = synchronize(merged)
    
    with timer("feature engineering"):
        engineered = engineer_features(synchronized)

    with timer("compute train split"):
        train_run_ids = _get_train_run_ids(engineered)

    with timer("normalization"):
        normalized = normalize(
            engineered,
            train_global_run_ids=train_run_ids,
            method=CONFIG.norm_method,
        )
        feature_cols = get_feature_cols(normalized)

    with timer("windows"):
        X, labels = create_temporal_windows(normalized, feature_cols)
    
    with timer("save dataset"):
        save_dataset(train_run_ids=train_run_ids)

    log("Preprocessing complete — check processed/final/ for train/val/test.npz")

if __name__ == "__main__":
    run_preprocessing()