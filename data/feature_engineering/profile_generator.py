import pandas as pd

def generate_conservative_profile(summary_df): # cautious highway cruising
    conservative = summary_df[
        (summary_df["steady_velocity"] < 20) &
        (summary_df["steady_spacing"] > 25) &
        (summary_df["max_acceleration"] < 0.8)
    ]
    return build_profile(conservative, "Conservative")

def generate_stable_profile(summary_df): # stable highway platooning
    stable = summary_df[
        (summary_df["steady_velocity"] >= 20) &
        (summary_df["steady_velocity"] <= 25) &
        (summary_df["steady_spacing"] >= 10) &
        (summary_df["steady_spacing"] <= 20)
    ]

    return build_profile(stable, "Normal Highway")

def generate_aggressive_profile(summary_df): # highway merge
    aggressive = summary_df[
        (summary_df["max_accel_magnitude"] > 1.5) |
        (summary_df["rms_jerk"] > 2.0)
    ]

    return build_profile(aggressive, "Aggressive Merge")

def generate_congested_profile(summary_df): # dense highway traffic
    congested = summary_df[
        (summary_df["steady_velocity"] < 12)
    ]

    return build_profile(congested, "Congested")

def generate_deceleration_profile(summary_df): # highway braking
    braking = summary_df[
        (summary_df["min_acceleration"] < -1.5)
    ]

    return build_profile(braking, "Deceleration Event")

def build_profile(df, profile_name):
    if len(df) == 0:
        return {
            "profile": profile_name,
            "num_sequences": 0
        }
    
    return {
        "profile": profile_name,
        "num_sequences": len(df),
        "cruise_speed": df["steady_velocity"].mean(),
        "steady_spacing": df["steady_spacing"].mean(),
        "initial_spacing": df["initial_spacing"].mean(),
        "rise_rate": df["rise_rate"].mean(),
        "fall_rate": df["fall_rate"].mean(),
        "convergence_time": df["convergence_time"].mean(),
        "velocity_std": df["velocity_std"].mean(),
        "spacing_std": df["spacing_std"].mean(),
        "accel_std": df["accel_std"].mean(),
        "jerk_std": df["jerk_std"].mean()
    }

def generate_all_profiles(summary_df):
    profiles = []

    profiles.append(generate_conservative_profile(summary_df))
    profiles.append(generate_stable_profile(summary_df))
    profiles.append(generate_aggressive_profile(summary_df))
    profiles.append(generate_congested_profile(summary_df))
    profiles.append(generate_deceleration_profile(summary_df))

    return pd.DataFrame(profiles)