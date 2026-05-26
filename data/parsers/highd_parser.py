from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2] # goes up 3 levels from script
DATA_DIR = BASE_DIR / "data" / "raw" / "highD"

def load_recording(recording_id):
    ''' Load a single highD recording by its ID. '''
    tracks_path = DATA_DIR / f"{recording_id:02d}_tracks.csv"
    meta_path = DATA_DIR / f"{recording_id:02d}_tracksMeta.csv"

    tracks = pd.read_csv(tracks_path)
    meta = pd.read_csv(meta_path)

    tracks["recordingId"] = recording_id
    meta["recordingId"] = recording_id

    return tracks, meta


def load_all_recordings():
    ''' Load all 60 highD recordings. '''

    all_tracks = []
    all_meta = []

    for recording_id in range(1, 61):
        print(f"Loading recording {recording_id:02d}")

        tracks, meta = load_recording(recording_id)
        all_tracks.append(tracks)
        all_meta.append(meta)

    tracks_df = pd.concat(all_tracks, ignore_index=True)
    meta_df = pd.concat(all_meta, ignore_index=True)

    return tracks_df, meta_df
    
def filter_trucks(tracks_df, meta_df):
    ''' Keep only truck trajectories. '''
    
    truck_meta = meta_df[meta_df["class"] == "Truck"]
    truck_ids = truck_meta["id"].unique()
    truck_tracks = tracks_df[
        tracks_df["id"].isin(truck_ids)
    ].copy()

    return truck_tracks, truck_meta

def keep_longitudinal_columns(tracks_df):
    ''' Keep only longitudinal platooning features. '''

    columns = [
        "recordingId",
        "frame",
        "id",
        "laneId",
        "precedingId",
        "followingId",
        "x",
        "xVelocity",
        "xAcceleration",
        "dhw", # distance headway
        "thw", # time headway
        "ttc" # time to collision
    ]

    return tracks_df[columns].copy()

if __name__ == "__main__":
    tracks_df, meta_df = load_all_recordings()

    print("Tracks shape:", tracks_df.shape)
    print("Meta shape:", meta_df.shape)

    trucks_tracks, trucks_meta = filter_trucks(tracks_df, meta_df)

    truck_tracks, truck_meta = filter_trucks(tracks_df, meta_df)
    print("Truck tracks:", truck_tracks.shape)
    print("Truck vehicles:", len(truck_meta))