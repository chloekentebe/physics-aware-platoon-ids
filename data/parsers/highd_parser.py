import pandas as pd

TRACKS_PATH = "data/raw/highD/01_tracks.csv"
META_PATH = "data/raw/highD/01_tracksMeta.csv"

def load_highd():
    tracks = pd.read_csv(TRACKS_PATH)
    tracks_meta = pd.read_csv(META_PATH)
    
    return tracks, tracks_meta

if __name__ == "__main__":
    tracks, tracks_meta = load_highd()
    print(tracks.head())
    print(tracks_meta.head())