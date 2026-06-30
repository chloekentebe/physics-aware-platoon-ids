'''Purpose: load windowed data, encode hierarchical labels, split by GlobalRunID
and save train/val/test .npz files.

Split strategy: RunID-based + stratified by (Profile, attack_type)
    - RunID-based: all windows from a given run go to exactly ONE split.
      This prevents leakage where overlapping windows from the same run appear on
      both side of a split, which would inflate test accuracy artificially.
    - Stratified: each (Profile, attack_type) combination is proportionally
      represented in train/val/test. With ~10 runs per combo, a naive random split
      could easily zero out a class in one split.

'''

import numpy as np
from pathlib import Path

from config import CONFIG
from helpers.io import save_json, load_json
from helpers.labels import encode_hierarchy, build_all_label_maps
from helpers.utils import ensure_dir, set_seed, log
from collections import Counter

def stratified_run_split(run_metadata: dict, seed: int) -> dict[str, set]:
    '''
    splits GlobalRunIDs into train/val/test stratified by (Profile, attack_type)

    run_metadata: {GlobalRunID: {"Profile": ..., "attack_type": ...}, ...}
    Returns:      {"train": {run_id, ...}, "val": {...}, "test": {...}}
    '''
    rng = np.random.default_rng(seed)

    # group run IDs by their stratum
    strata: dict[tuple, list] = {}
    for run_id, meta in run_metadata.items():
        key = (meta["Profile"], meta["attack_type"])
        strata.setdefault(key, []).append(run_id)
    
    splits = {"train": set(), "val": set(), "test": set()}

    log("Stratified split by (Profile, attack_type):")
    log(f"{'Profile':<25} {'AttackType':<14} {'N':>4} {'train':>6} {'val':>5} {'test:'>5}")

    for (profile, attack_type), run_ids in sorted(strata.items()):
        run_ids = list(run_ids)
        rng.shuffle(run_ids)
        n = len(run_ids)

        n_train = max(1, round(n * CONFIG.train_fraction))
        n_val   = max(1, round(n * CONFIG.val_fraction)) if n >= 3 else 0
        n_test = n - n_train - n_val

        if n_test < 0:
            n_train, n_val, n_test = n, 0, 0
        
        splits["train"].update(run_ids[:n_train])
        splits["val"].update(run_ids[n_train:n_train + n_val])
        splits["test"].update(run_ids[n_train + n_val:])

        log(f"{profile:<25} {attack_type:<14} {n:>4} {n_train:>6} {n_val:>5} {n_test:>5}")

    # zero-overlap check
    assert not (splits["train"] & splits["val"], "train/val GlobalRunID overlap!")
    assert not (splits["train"] & splits["test"], "Train/test GlobalRunID overlap!")
    assert not (splits["val"] & splits["test"], "Val/test GlobalRunID overlap!")

    log(f"Split totals -> train: {len(splits['train'])} runs | "
        f"val: {len(splits['val'])} runs | test: {len(splits['test'])} runs")
    return splits

def compute_class_weights(labels: np.ndarray) -> dict:
    '''inverse frequency weighting for imbalanced classes'''
    counts = Counter(labels.tolist())
    total = sum(counts.values())
    return {cls: total / (len(counts) * count)
            for cls, count in counts.items()}


def save_dataset():
    '''
    loads windowed data, encodes the full label hierarchy, performs a
    RunID-stratified split, and writes one .npz per split plus metadata
    '''
    # load windowed data
    windows_path = CONFIG.windows_dir / CONFIG.windows_file
    log(f"Loading windows from {windows_path}")
    data = np.load(windows_path, allow_pickle=True)

    X               = data["X"]                             # [N, W, F]
    global_run_ids  = data["global_run_ids"].astype(str)    # [N]
    profiles        = data["profiles"].astype(str)          # [N] 
    attack_types    = data["label_attack_type"].astype(str) # [N] raw string from windows

    n = len(X)
    log(f"Loaded {n:,} windows, feature shape: {X.shape[1:]}")

    # encode hierarchical labels
    # windows.py store raw string labels - encode_hierarchy converts them to
    # four parallel integer arrays covering every level of the tree
    raw_labels = {
        "is_attack":        data["label_is_attack"].tolist(),
        "attack_vector":    data["label_attack_vector"].astype(str).tolist(),
        "attacker_id":      data["label_attacker_id"].astype(str).tolist(),
        "attack_type":    data["label_attack_type"].astype(str).tolist(),
    }
    encoded = encode_hierarchy(raw_labels)

    class_weights = {
        "is_attack":     compute_class_weights(encoded["label_is_attack"]),
        "attack_vector": compute_class_weights(encoded["label_attack_vector"]),
        "attacker_id":   compute_class_weights(encoded["label_attacker_id"]),
        "attack_type":   compute_class_weights(encoded["label_attack_type"]),
    }
    save_json(class_weights, CONFIG.final_dir / "class_weights.json")

    # build run metadata for stratified split
    # one entry per unique GlobalRunID (profile and attack_type are constant
    # within a run so we just take the first window that belongs to each run)
    run_metadata = {}
    for i, run_id in enumerate(global_run_ids):
        if run_id not in run_metadata:
            run_metadata[run_id] = {
                "Profile": profiles[i],
                "attack_type": attack_types[i],
            }

    splits = stratified_run_split(run_metadata, seed=CONFIG.random_seed)

    # map window indices to split membership
    split_indices = {name: [] for name in splits}
    for i, run_id in enumerate(global_run_ids):
        for split_name, run_id_set in splits.items():
            if run_id in run_id_set:
                split_indices[split_name].append(i)
                break
    
    # save each split
    ensure_dir(CONFIG.final_dir)

    for split_name, indices in split_indices.items():
        idx = np.array(indices)
        if len(idx) == 0:
            log(f"[WARN] {split_name} split is empty - chekc run counts per stratum")
            continue

        output = {
            "X":                X[idx],
            "global_run_ids":   global_run_ids[idx],
        }

        # add every hierarchy level as a separate array
        for label_key, label_arr in encoded.items():
            output[label_key] = label_arr[idx]
        
        out_path = CONFIG.final_dir / f"{split_name}.npz"
        np.savez_compressed(out_path, **output)

        n_attack = int((output["label_is_attack"] == 1).sum())
        n_normal = int((output["label_is_attack"] == 0).sum())
        log(f"Saved {split_name}.npz -> {len(idx):,} windows "
            f"({n_attack:,} attack / {n_normal:,} normal)")
    
    # save label maps and metadata
    label_maps = build_all_label_maps()
    save_json(label_maps, CONFIG.final_dir / "label_maps.json")

    metadata = {
        "n_windows":    n,
        "window_size":  int(CONFIG.window_size),
        "stride":       int(CONFIG.stride),
        "n_features":   int(X.shape[2]),
        "split_sizes":  {name: len(idx) for name, idx in split_indices.items()},
        "hierarchy": {
            "level_0": "label_is_attack     (0=Normal, 1=Attack)",
            "level_1": "label_attack_vector (0=None, 1=V2V, 2=V2I)",
            "level_2": "label_attacker_id  (0=None, 1=RSU, 2=Leader, 3-6=Follower1-4)",   
            "level_3": "label_attack_type   (0=None, 1=SpeedFDI,...,6=TimingFDI)",
        }
    }

    save_json(metadata, CONFIG.final_dir / "dataset_metadata.json")
    log(f"Saved label_maps.json and dataset_metadata.json to {CONFIG.final_dir}")

if __name__ == "__main__":
    save_dataset()