'''purpose: combine the raw CSV file
input --> platoon csv + bsm csv
output --> merged dataframe

'''

import re
import pandas as pd
from config import CONFIG
from helpers.io import list_csv_files, load_csv, save_dataframe
from helpers.utils import ensure_dir, log
from helpers.validation import require_non_empty, require_columns

def _source_type(path):
    return "platoon" if path.name.startswith("platoon") else "bsm"

def _run_from_name(path):
    match = re.search(r"(?:run_)?(\d+)(?=\.csv$)", path.name)
    return int(match.group(1)) if match else None

def _load_folder(folder, dataset):
    frames = []

    for path in list_csv_files(folder):
        df = load_csv(path)
        require_non_empty(df, str(path))

        df["source_file"] = path.name
        df["source_type"] = _source_type(path)
        df["dataset"] = dataset

        if "RunID" not in df.columns:
            inferred_run = _run_from_name(path)
            if inferred_run is not None:
                df["RunID"] = inferred_run

        
        if "ScenarioType" not in df.columns:
            df["ScenarioType"] = "Attack" if dataset in {"v2v", "v2i"} else "Normal"
        
        df["is_attack"] = df["IsAttack"] if "IsAttack" in df.columns else 0
        df["attack_vector"] = df["AttackVector"] if "AttackVector" in df.columns else "None"
        df["attack_type"] = df["AttackType"] if "AttackType" in df.columns else "None"
        df["attacker_id"]   = df["AttackerID"]   if "AttackerID"   in df.columns else "Normal"
        df["attack_active"] = df["AttackActive"] if "AttackActive" in df.columns else 0

        frames.append(df)

    return pd.concat(frames, ignore_index=True, sort=False)

def merge():
    normal = _load_folder(CONFIG.normal_dir, "normal")
    v2i = _load_folder(CONFIG.v2i_dir, "v2i")
    v2v = _load_folder(CONFIG.v2v_dir, "v2v")

    merged = pd.concat([normal, v2v, v2i], ignore_index=True, sort=False)
    require_columns(merged, ["RunID", "source_type", "dataset"], "merged data")

    if "GlobalRunID" not in merged.columns:
        merged["GlobalRunID"] = (
            merged["dataset"].astype(str) + "__" +
            merged.get("Profile", merged.get("profile", "unknown")).astype(str) + "__" +
            merged["attack_type"].astype(str) + "__" +
            merged["RunID"].astype(str)
        )

    ensure_dir(CONFIG.merged_dir)
    out_path = CONFIG.merged_dir / CONFIG.merged_file
    save_dataframe(merged, out_path)


    # log run distributions to verify counts before syncing
    run_counts = (
        merged.drop_duplicates("GlobalRunID")
        .groupby(["dataset", "attack_type"])
        .size()
    )
    log(f"Run distribution:\n{run_counts.to_string()}")
    log(f"Saved merged data -> {out_path} ({len(merged):,} rows)")
    return merged

if __name__ == "__main__":
    merge()