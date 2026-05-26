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
        "min_spacing": spacing.min(),
        "max_spacing": spacing.max(),
        "spacing_std": spacing.std()
    }

def compute_velocity_statistics(sequence):
    ''' Compute velcity metrics. '''

    velocity = sequence["xVelocity"].abs()

    return {
        "initial_velocity": velocity.iloc[0],
        "steady_velocity": velocity.tail(50).mean(),
        "max_velocity": velocity.max(),
        "velocity_std": velocity.std()
    }

def compute_acceleration_statistics(sequence):
    ''' Compute acceleration dynamics. '''

    accel = sequence["xAcceleration"]
    accel_abs = accel.abs()
    dt = 1 / 25
    jerk = accel.diff().fillna(0) / dt

    return {
        "max_acceleration": accel.max(),
        "min_acceleration": accel.min(),

        "mean_acceleration": accel.mean(),
        "accel_std": accel.std(),
        "max_accel_magnitude": accel_abs.max(),
        "mean_accel_magnitude": accel_abs.mean(),

        "rise_rate": jerk.max(), # max acceleration rate
        "fall_rate": jerk.min(), # min acceleration rate

        "mean_jerk": jerk.mean(),
        "jerk_std": jerk.std()
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

def compute_relative_velocity_statistics(sequence):
    ''' Compute leader-follower relative velocity statistics. '''
    
    leader_id = sequence["leaderId"].iloc[0]
    follower_id = sequence["followerId"].iloc[0]

    leader = sequence[sequence["id"] == leader_id]
    follower = sequence[sequence["id"] == follower_id]

    merged = follower.merge(
        leader,
        on="frame",
        suffixes=("_f", "_l")
    )

    relative_velocity = merged["xVelocity_l"] - merged["xVelocity_f"]

    return {
        "mean_relative_velocity": relative_velocity.mean(),
        "max_relative_velocity": relative_velocity.max(),
        "min_relative_velocity": relative_velocity.min(),
        "relative_velocity_std": relative_velocity.std()
    }

def compute_relative_acceleration_statistics(sequence):
    ''' Compute leader-follower relative acceleration statistics. '''
    
    leader_id = sequence["leaderId"].iloc[0]
    follower_id = sequence["followerId"].iloc[0]

    leader = sequence[sequence["id"] == leader_id]
    follower = sequence[sequence["id"] == follower_id]

    merged = follower.merge(
        leader,
        on="frame",
        suffixes=("_f", "_l")
    )

    relative_accel = merged["xAcceleration_l"] - merged["xAcceleration_f"]

    return {
        "mean_relative_acceleration": relative_accel.mean(),
        "max_relative_acceleration": relative_accel.max(),
        "min_relative_acceleration": relative_accel.min(),
        "relative_acceleration_std": relative_accel.std()
    }

def compute_headway_consistency(sequence):
    ''' Compute time headway stability metrics. '''

    thw = sequence["thw"]

    return {
        "mean_thw": thw.mean(),
        "std_thw": thw.std(),
        "min_thw": thw.min(),
        "max_thw": thw.max()
    }


def compute_string_stability_metrics(sequence):
    ''' Compute platoon string stability indicators. '''

    velocity = sequence["xVelocity"].abs()
    accel = sequence["xAcceleration"]

    velocity_oscillation = velocity.std()
    accel_oscillation = accel.std()

    spacing = sequence["dhw"]
    spacing_oscillation = spacing.std()

    return {
        "velocity_oscillation": velocity_oscillation,
        "accel_oscillation": accel_oscillation,
        "spacing_oscillation": spacing_oscillation
    }

def compute_smoothness_metrics(sequence):
    accel = sequence["xAcceleration"]
    jerk = accel.diff().fillna(0) / (1/25)

    return {
        "rms_acceleration": np.sqrt(np.mean(accel**2)),
        "rms_jerk": np.sqrt(np.mean(jerk**2))
    }

def summarize_sequence(sequence):
    ''' Create a single summary row for a sequence. '''

    spacing_stats = compute_spacing_statistics(sequence)
    velocity_stats = compute_velocity_statistics(sequence)
    accel_stats = compute_acceleration_statistics(sequence)
    relative_velocity_stats = compute_relative_velocity_statistics(sequence)
    relative_accel_stats = compute_relative_acceleration_statistics(sequence)
    headway_stats = compute_headway_consistency(sequence)
    string_stability_stats = compute_string_stability_metrics(sequence)
    smoothness_stats = compute_smoothness_metrics(sequence)

    summary = {
        "leader_id": sequence["leaderId"].iloc[0],
        "follower_id": sequence["followerId"].iloc[0],
        "duration_frames": len(sequence),
        "convergence_time": estimate_convergence_time(sequence)
    }

    summary.update(spacing_stats)
    summary.update(velocity_stats)
    summary.update(accel_stats)
    summary.update(relative_velocity_stats)
    summary.update(relative_accel_stats)
    summary.update(headway_stats)
    summary.update(string_stability_stats)
    summary.update(smoothness_stats)

    return summary

def build_sequence_summary_dataframe(sequences):
    ''' Build summary tale for all sequences. '''

    summaries = []
    for seq in sequences:
        summaries.append(summarize_sequence(seq))
    
    return pd.DataFrame(summaries)