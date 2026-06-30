'''contains every reusable physics calculation'''
import numpy as np

def safe_divide(numerator, denominator, default=0.0):
    return np.where(np.abs(denominator) > 1e-9, numerator/denominator, default)

def difference(a, b):
    return a - b

def absolute_error(a, b):
    return np.abs(a - b)

def relative_speed(front_speed, rear_speed):
    return front_speed - rear_speed

def relative_acceleration(front_accel, rear_accel):
    return front_accel - rear_accel

def closing_rate(front_speed, rear_speed):
    '''
    positive = rear vehicle catching up to fron vehicle (gap shrinking)
    negative = rear vehicle falling behind (gap growing)
    '''
    return rear_speed - front_speed

def time_headway(spacing, speed):
    return safe_divide(spacing, speed)

def spacing_error(actual_spacing, desired_spacing):
    return actual_spacing - desired_spacing

def speed_error(actual_speed, desired_speed):
    return actual_speed - desired_speed

def predicted_spacing(previous_spacing, relative_speed_value, dt):
    return previous_spacing + relative_speed_value * dt

def predicted_speed(previous_speed, relative_acceleration_value, dt):
    return previous_speed + relative_acceleration_value * dt

def residual(actual, predicted):
    return actual - predicted

def jerk(acceleration, dt):
    if hasattr(acceleration, "to_numpy"):
        acceleration = acceleration.to_numpy()
    diff = np.diff(acceleration, prepend=acceleration[0])
    return safe_divide(diff, dt)

def rolling_rms(series, window):
    if not hasattr(series, "rolling"):
        series = pd.Series(series)
    return np.sqrt(series.pow(2).rolling(window, min_periods=1).mean())
