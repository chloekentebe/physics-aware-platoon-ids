import pandas as pd
import numpy as np

MIN_SEQUENCE_LENGTH = 200

def extract_following_sequences(truck_tracks):
    ''' Extract stable truck-following structures. '''

    sequences = []
    grouped = truck_tracks.groupby("id")

    for vehicle_id, vehicle_df in grouped:
        vehicle_df = vehicle_df.sort_values("frame")
        preceding_ids = vehicle_df["precedingId"].unique()
        
        for preceding_id in preceding_ids:
            if preceding_id == 0:
                continue

            seq = vehicle_df[
                vehicle_df["precedingId"] == preceding_id
            ].copy()

            if len(seq) < MIN_SEQUENCE_LENGTH:
                continue

            lane_changes = seq["laneId"].nunique()
            if lane_changes > 1:
                continue

            seq["followerId"] = vehicle_id
            seq["leaderId"] = preceding_id

            sequences.append(seq)

        return sequences
    
def compute_spacing_statistics(sequence):
    ''' Compute spacing metrics. '''

    spacing = sequence["dhw"]

    return {
        "initial_spacing": spacing.iloc[0],
        "steady_spacing": spacing.tail(50).mean(),
        "min_spacing:": spacing.min(),
        "max_spacing": spacing.max(),
        "spacing_std": spacing.std()
    }

def compute_velocity_statistics(sequence):
    ''' Compute velcity metrics. '''

    velocity = sequence["xVelocity"]

    return {
        "initial_velocity": velocity.iloc[0],
        "steady_velocity": velocity.tail(50).mean,
        "max_velocity": velocity.max(),
        "velocity_std": velocity.std()
    }

def compute_acceleration_statistics(sequence):
    ''' Compute acceleration dynamics. '''

    accel = sequence["xAcceleration"]

    return {
        "max_acceleration": accel.max(),
        "min_acceleration": accel.min(),
        "mean_acceleration": accel.mean(),
        "accel_std": accel.std(),
        "rise_rate:": np.percentile(accel, 95), # max acceleration
        "fall_rate:": np.percetile(accel, 5) # min acceleration
    }

def estimate_convergence_time(sequence, threshold=2.0):
    ''' Estimate convergence time to steady spacing. '''

    spacing = sequence["dhw"].values

    steady_state = np.mean(spacing[-50:])
    error = np.abs(spacing - steady_state)
    indices = np.where(error < threshold)[0]

    if len(indices) == 0:
        return np.nan
    
    first_idx = indices[0]
    fps = 25

    return first_idx / fps

def summarize_sequence(sequence):
    ''' Create a single summary row for a sequence. '''

    spacing_stats = compute_spacing_statistics(sequence)
    velocity_stats = compute_velocity_statistics(sequence)
    accel_stats = compute_acceleration_statistics(sequence)

    summary = {
        "leader_id": sequence["leaderId"].iloc[0],
        "follower_id": sequence["followerId"].iloc[0],
        "duration_frames": len(sequence),
        "convergence_time": estimate_convergence_time(sequence)
    }

    summary.update(spacing_stats)
    summary.update(velocity_stats)
    summary.update(accel_stats)

    return summary

def build_sequence_summary_dataframe(sequences):
    ''' Build summary tale for all sequences. '''

    summaries = []
    for seq in sequences:
        summaries.append(summarize_sequence(seq))
    
    return pd.DataFrame(summaries)