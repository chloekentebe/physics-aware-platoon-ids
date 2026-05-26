import pandas as pd
import numpy as np

PLATOON_SPACING_OVERRIDE = {
    "Conservative":       {"steady_spacing": 25.0, "initial_spacing": 40.0},
    "Normal Highway":     {"steady_spacing": 15.0, "initial_spacing": 25.0},
    "Aggressive Merge":   {"steady_spacing": 10.0, "initial_spacing": 20.0},
    "Congested":          {"steady_spacing": 8.0,  "initial_spacing": 15.0},
    "Deceleration Event": {"steady_spacing": 12.0, "initial_spacing": 30.0},
}

def build_profile(df, profile_name, override_spacing=True):
    if len(df) == 0:
        return {"profile": profile_name, "num_sequences": 0}

    profile = {
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

    # override spacing with realistic platoon values
    # keep dynamics (rise_rate, fall_rate, jerk) from real highD data
    if override_spacing and profile_name in PLATOON_SPACING_OVERRIDE:
        profile.update(PLATOON_SPACING_OVERRIDE[profile_name])

    return profile

def compute_quantiles(summary_df):
    ''' Compute dataset-relative thresholds. '''
    return {
        "v25": summary_df["steady_velocity"].quantile(0.25), 
        "v75": summary_df["steady_velocity"].quantile(0.75),  
        "s25": summary_df["steady_spacing"].quantile(0.25),   
        "s50": summary_df["steady_spacing"].quantile(0.50),   
        "s75": summary_df["steady_spacing"].quantile(0.75),   
        "j75": summary_df["rms_jerk"].quantile(0.75),        
        "j90": summary_df["rms_jerk"].quantile(0.90),
        "a75": summary_df["max_accel_magnitude"].quantile(0.75),
        "a90": summary_df["max_accel_magnitude"].quantile(0.90),
    }

def generate_conservative_profile(summary_df, q):
    ''' Slow speed, large spacing, smooth driving. '''
    df = summary_df[
        (summary_df["steady_velocity"] < q["v25"]) &
        (summary_df["steady_spacing"] > q["s50"]) &
        (summary_df["rms_jerk"] < q["j75"])
    ]
    return build_profile(df, "Conservative")

def generate_normal_profile(summary_df, q):
    ''' Mid-range speed and spacing, typical highway cruising. '''
    df = summary_df[
        (summary_df["steady_velocity"] >= q["v25"]) &
        (summary_df["steady_velocity"] <= q["v75"]) &
        (summary_df["steady_spacing"] >= q["s25"]) &
        (summary_df["steady_spacing"] <= q["s75"]) &
        (summary_df["rms_jerk"] < q["j75"])
    ]
    return build_profile(df, "Normal Highway")

def generate_aggressive_profile(summary_df, q):
    ''' High jerk or acceleration — aggressive maneuvers. '''
    df = summary_df[
        (summary_df["max_accel_magnitude"] > q["a90"]) |
        (summary_df["rms_jerk"] > q["j90"])
    ]
    return build_profile(df, "Aggressive Merge")

def generate_congested_profile(summary_df, q):
    ''' Low speed, close spacing — congested conditions. '''
    df = summary_df[
        (summary_df["steady_velocity"] < q["v25"]) &
        (summary_df["steady_spacing"] < q["s25"])
    ]
    return build_profile(df, "Congested")

def generate_deceleration_profile(summary_df, q):
    ''' Strong braking events — fall_rate in bottom 10%. '''
    fall_rate_10 = summary_df["fall_rate"].quantile(0.10)  # most negative
    df = summary_df[
        (summary_df["fall_rate"] < fall_rate_10) &
        (summary_df["rms_jerk"] > q["j75"])
    ]
    return build_profile(df, "Deceleration Event")

def generate_all_profiles(summary_df):
    q = compute_quantiles(summary_df)

    print("Quantile thresholds:")
    for k, v in q.items():
        print(f"  {k}: {v:.3f}")

    profiles = []
    profiles.append(generate_conservative_profile(summary_df, q))
    profiles.append(generate_normal_profile(summary_df, q))
    profiles.append(generate_aggressive_profile(summary_df, q))
    profiles.append(generate_congested_profile(summary_df, q))
    profiles.append(generate_deceleration_profile(summary_df, q))

    profiles_df = pd.DataFrame([p for p in profiles if p["num_sequences"] > 0])

    total = profiles_df["num_sequences"].sum()
    print(f"\nSequences captured: {total} / {len(summary_df)} ({100*total/len(summary_df):.1f}%)")

    return profiles_df