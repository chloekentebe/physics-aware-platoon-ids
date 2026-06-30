'''Purpose: load every raw CSV (platoon + BSM) from normal/v2v/v2i folders,
assign a collision-free GlobalRunID derived from the filename,
validate that every run has both a platoon and BSM file, and concatenate
everything into one merged DataFrame ready for synchronization

Naming conventions handled (from MATLAB generation scripts):

  NORMAL runs — platoon and BSM use DIFFERENT numbering:
    platoon_run_{loopIdx+offset}.csv          (offset: conservative=0,
    normal_{profile}_bsm_run_{loopIdx}.csv     stable=30, congested=60,
                                               deceleration=90)

  V2V attack runs — platoon and BSM use SAME numbering:
    platoon_{type}fdi_{profile}_{loopIdx}.csv
    bsm_{type}fdi_{profile}_{loopIdx}.csv
    type ∈ {acc, mc, pos, speed}

  V2I attack runs — same pattern as V2V:
    platoon_{type}fdi_{profile}_{loopIdx}.csv
    bsm_{type}fdi_{profile}_{loopIdx}.csv
    type ∈ {content, timing}

The GlobalRunID is built as '{dataset}__{profile}__{attack_type}__{loopIdx}'
where loopIdx is ALWAYS extracted from the BSM filename (the stable index),
and the platoon file is matched to its BSM partner by (profile, attack_type,
loopIdx) rather than by the raw numeric suffix in the platoon filename. 
'''

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import CONFIG
from utils import log, ensure_dir
from validation import require_columns, require_non_empty

# PROFILE AND ATTACK TYPE VOCABULARY
PROFILES = {
    "conservative": "Conservative",
    "stable":       "Stable",
    "congested":    "Congested",
    "deceleration": "Deceleration",
}

V2V_ATTACK_TYPES = {"acc": "AccFDI", "mc": "MsgCntFDI",
                     "pos": "PosFDI", "speed": "SpeedFDI"}
V2I_ATTACK_TYPES = {"content": "ContentFDI", "timing": "TimingFDI"}
ALL_ATTACK_TYPES = {**V2V_ATTACK_TYPES, **V2I_ATTACK_TYPES}

# FILE METADATA
@dataclass
class FileInfo:
    path:        Path
    file_type:   str          # "platoon" or "bsm"
    dataset:     str          # "normal", "v2v", "v2i"
    profile:     str          # "Conservative", "Stable", "Congested", "Deceleration"
    attack_type: str          # "None", "SpeedFDI", "AccFDI", etc.
    loop_idx:    int          # the raw loopIdx from MATLAB (from BSM filename)
    global_run_id: str        # collision-free join key

def _parse_normal_bsm(name: str, path: Path) -> Optional[FileInfo]:
    '''
    matches: normal_{profile}_bsm_run_{N}.csv
    this is the STABLE index (loopIdx without any offset)
    '''
    m = re.fullmatch(r"normal_([a-z]+)_bsm_run_(\d+)", name)
    if not m:
        return None
    profile_key, idx = m.group(1), int(m.group(2))
    profile = PROFILES.get(profile_key)
    if profile is None:
        return None
    gid = f"normal__{profile_key}__None__{idx}"
    return FileInfo(path, "bsm", "normal", profile, "None", idx, gid)

def _parse_normal_platoon(name: str, path: Path) -> Optional[FileInfo]:
    '''
    matches: platoon_run_{N}.csv  (N includes the MATLAB offset)
    cannot know the loopIdx from this filename alone --> it will be
        matched to its BSM partner via folder context in _pair_normal_files()
    returns None here; normal platoon files are handled separately
    '''
    m = re.fullmatch(r"platoon_run_(\d+)", name)
    if not m:
        return None
    # Store the raw suffix as loop_idx placeholder; real matching done later
    return FileInfo(path, "platoon", "normal", "unknown", "None",
                    int(m.group(1)), "")

def _parse_attack_file(name: str, path: Path, dataset: str) -> Optional[FileInfo]:
    '''
    matches both V2V and V2I attack files:
      platoon_{type}fdi_{profile}_{N}.csv
      bsm_{type}fdi_{profile}_{N}.csv
    '''
    file_type = "platoon" if name.startswith("platoon_") else "bsm"
    m = re.fullmatch(
        rf"{file_type}_([a-z]+)fdi_([a-z]+)_(\d+)", name
    )
    if not m:
        return None
    type_key, profile_key, idx = m.group(1), m.group(2), int(m.group(3))
    profile = PROFILES.get(profile_key)
    attack_type = ALL_ATTACK_TYPES.get(type_key)
    if profile is None or attack_type is None:
        return None
    gid = f"{dataset}__{profile_key}__{type_key}__{idx}"
    return FileInfo(path, file_type, dataset, profile, attack_type, idx, gid)

# NORMAL-RUN PLATOON<->BSM PAIRING
# normal files cannot be matched by number alone because of matlab offsets
# instead BSM files are loaded first (stable index), then each platoon file is matched
# to its BSM partner by folder position within the same profile folder

# platoon file offset per profile 
NORMAL_PLATOON_OFFSET = {
    "conservative": 0,
    "stable":       30,
    "congested":    60,
    "deceleration": 90,
}

def _pair_normal_files(normal_dir: Path) -> list[tuple[FileInfo, FileInfo]]:
    '''
    for each normal BSM file, find its corresponding platoon file using the
    known MATLAB offset: platoon_run_{loopIdx + offset}.csv

    returns a list of (platoon_info, bsm_info) pairs
    '''
    pairs = []
    unmatched_bsm = []

    all_files = list(normal_dir.rglob("*.csv"))

    # parse BSM files first as these carry the stable loopIdx
    bsm_infos = []
    for p in all_files:
        name = p.stem.lower()
        info = _parse_normal_bsm(name, p)
        if info:
            bsm_infos.append(info)

    # build a lookup of platoon files by their raw numeric suffix
    platoon_by_suffix: dict[int, Path] = {}
    for p in all_files:
        name = p.stem.lower()
        m = re.fullmatch(r"platoon_run_(\d+)", name)
        if m:
            platoon_by_suffix[int(m.group(1))] = p

    # match each BSM to its platoon partner
    for bsm_info in bsm_infos:
        profile_key = bsm_info.profile.lower()
        offset = NORMAL_PLATOON_OFFSET.get(profile_key, 0)
        platoon_suffix = bsm_info.loop_idx + offset
        platoon_path = platoon_by_suffix.get(platoon_suffix)

        if platoon_path is None:
            log(f"[WARN] No platoon file for BSM {bsm_info.path.name} "
                f"(expected platoon_run_{platoon_suffix}.csv)")
            unmatched_bsm.append(bsm_info.path.name)
            continue

        platoon_info = FileInfo(
            path=platoon_path,
            file_type="platoon",
            dataset="normal",
            profile=bsm_info.profile,
            attack_type="None",
            loop_idx=bsm_info.loop_idx,   # use BSM's stable loopIdx
            global_run_id=bsm_info.global_run_id,
        )
        pairs.append((platoon_info, bsm_info))

    if unmatched_bsm:
        log(f"[WARN] {len(unmatched_bsm)} normal BSM files had no matching platoon file")
    log(f"Paired {len(pairs)} normal runs (platoon + BSM)")
    return pairs

def _pair_attack_files(folder: Path, dataset: str) -> list[tuple[FileInfo, FileInfo]]:
    '''
    for attack files, platoon and BSM share the same loopIdc so matching is straightforward:
        group by (profile, attack_type, loop_idx)
    '''
    platoon_map: dict[str, FileInfo] = {}
    bsm_map:     dict[str, FileInfo] = {}

    for p in folder.rglob("*.csv"):
        name = p.stem.lower()
        info = _parse_attack_file(name, p, dataset)
        if info is None:
            continue
        key = f"{info.profile}__{info.attack_type}__{info.loop_idx}"
        if info.file_type == "platoon":
            platoon_map[key] = info
        else:
            bsm_map[key] = info

    pairs = []
    for key, bsm_info in bsm_map.items():
        platoon_info = platoon_map.get(key)
        if platoon_info is None:
            log(f"[WARN] No platoon match for attack BSM: {bsm_info.path.name}")
            continue
        pairs.append((platoon_info, bsm_info))

    unmatched_platoon = set(platoon_map) - set(bsm_map)
    if unmatched_platoon:
        log(f"[WARN] {len(unmatched_platoon)} {dataset} platoon files had no BSM match")

    log(f"Paired {len(pairs)} {dataset} runs (platoon + BSM)")
    return pairs

# CSV loading and label injection
def _load_pair(platoon_info: FileInfo, bsm_info: FileInfo) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    loads one (platoon, BSM) pair, injects GlobalRunID and standardized attack-label columns
    into both DataFrames
    '''

    platoon_df = pd.read_csv(platoon_info.path)
    bsm_df     = pd.read_csv(bsm_info.path)

    require_non_empty(platoon_df, str(platoon_info.path))
    require_non_empty(bsm_df,     str(bsm_info.path))

    is_attack = int(platoon_info.dataset in {"v2v", "v2i"})

    for df in [platoon_df, bsm_df]:
        df["GlobalRunID"]   = platoon_info.global_run_id
        df["source_type"]   = "platoon" if df is platoon_df else "bsm"
        df["dataset"]       = platoon_info.dataset
        df["Profile"]       = platoon_info.profile
        df["loop_idx"]      = platoon_info.loop_idx

        # Standardized attack labels — use CSV value if present, else default
        df["is_attack"]      = df.get("IsAttack",     pd.Series(is_attack, index=df.index))
        df["attack_vector"]  = df.get("AttackVector", pd.Series("None" if not is_attack else
                                                                  ("V2V" if platoon_info.dataset == "v2v"
                                                                   else "V2I"), index=df.index))
        df["attack_type"]    = df.get("AttackType",   pd.Series(platoon_info.attack_type, index=df.index))
        df["attacker_id"]    = df.get("AttackerID",   pd.Series(-2 if not is_attack else -1, index=df.index))
        df["attack_active"]  = df.get("AttackActive", pd.Series(0, index=df.index))


    platoon_df["source_file"] = platoon_info.path.name
    bsm_df["source_file"]     = bsm_info.path.name

    return platoon_df, bsm_df


def merge():
    '''
    loads, pairs, labels, and concatenates all raw CSVs into one merged DataFrame

    saves to CONFIG.merged_dir / CONFIG.merged_file
    '''
    all_platoon, all_bsm = [], []

    # normal runs
    normal_pairs = _pair_normal_files(CONFIG.normal_dir)
    for platoon_info, bsm_info in normal_pairs:
        p, b = _load_pair(platoon_info, bsm_info)
        all_platoon.append(p)
        all_bsm.append(b)

    # V2V attack runs
    v2v_pairs = _pair_attack_files(CONFIG.v2v_dir, "v2v")
    for platoon_info, bsm_info in v2v_pairs:
        p, b = _load_pair(platoon_info, bsm_info)
        all_platoon.append(p)
        all_bsm.append(b)

    # V2I attack runs
    v2i_pairs = _pair_attack_files(CONFIG.v2i_dir, "v2i")
    for platoon_info, bsm_info in v2i_pairs:
        p, b = _load_pair(platoon_info, bsm_info)
        all_platoon.append(p)
        all_bsm.append(b)

    platoon_merged = pd.concat(all_platoon, ignore_index=True, sort=False)
    bsm_merged     = pd.concat(all_bsm,     ignore_index=True, sort=False)

    # tag which table each row came from before final concat
    platoon_merged["table"] = "platoon"
    bsm_merged["table"]     = "bsm"

    merged = pd.concat([platoon_merged, bsm_merged], ignore_index=True, sort=False)

    require_columns(merged, ["GlobalRunID", "source_type", "dataset", "Profile"], "merged")

    # summary
    runs = merged.groupby(["dataset", "Profile", "attack_type"])["GlobalRunID"].nunique()
    log("Run counts by dataset / profile / attack type:")
    log(str(runs.to_string()))
    log(f"Total rows: {len(merged):,}  |  Total runs: {merged['GlobalRunID'].nunique()}")

    ensure_dir(CONFIG.merged_dir)
    out_path = CONFIG.merged_dir / CONFIG.merged_file
    merged.to_parquet(out_path.with_suffix(".parquet"), index=False)
    log(f"Saved merged data -> {out_path.with_suffix('.parquet')}")
  

if __name__ == "__main__":
    merge()