'''Purpose: scale numerical features for the BiLSTM
    Always normalize: spacings, speeds, acceleration, consistency features,
'''

from config import CONFIG
from helpers.io import load_csv, save_dataframe, save_json
from helpers.normalization import fit_scaler, transform, save_scaler
from helpers.utils import ensure_dir, log

def select_feature_columns(df):
    excluded = set(CONFIG.id_columns)
    excluded.update({"time", "source_type_bsm", "source_type_platoon"})
    numeric_cols = df.select_dtypes(include="number").columns
    return [col for col in numeric_cols if col not in excluded]

def normalize_features(engineered=None):
    if engineered is None:
        engineered = load_csv(CONFIG.engineered_dir / CONFIG.engineered_file)
    
    feature_cols = select_feature_columns(engineered)

    fit_df = engineered[engineered["dataset"] == "normal"]
    if fit_df.empty:
        fit_df = engineered
    
    scaler = fit_scaler(fit_df, feature_cols, method=CONFIG.normalization_method)
    normalized = transform(engineered, scaler, feature_cols)

    ensure_dir(CONFIG.normalized_dir)
    save_dataframe(normalized, CONFIG.normalized_dir / CONFIG.normalized_file)
    save_scaler(scaler, CONFIG.normalized_dir / "scaler.pkl")
    save_json(feature_cols, CONFIG.normalized_dir / "feature_names.json")

    log(f"saved normalized data -> {CONFIG.normalized_dir / CONFIG.normalized_file}")
    return normalized, feature_cols

if __name__ == "__main__":
    normalize_features()