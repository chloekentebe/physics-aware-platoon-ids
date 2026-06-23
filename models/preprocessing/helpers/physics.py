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
    return rear_speed - front_speed

def time_headway(spacing, speed):
    return safe_divide(spacing, speed)

def spacing_error(actual_spacing, desired_spacing):
    return actual_spacing - desired_spacing

def speed_error(actual_speed, desired_speed):
    return actual_speed - desired_speed

def position_error(actual_position, desired_position):
    return actual_position - desired_position

def acceleration_error(actual_acceleration, desired_acceleration):
    return actual_acceleration - desired_acceleration

def predicted_position(position, speed, dt):
    return position + (speed * dt)

def predicted_spacing(previous_spacing, relative_speed_value, dt):
    return previous_spacing + relative_speed_value * dt

def residual(actual, predicted):
    return actual - predicted

def jerk(acceleration, dt):
    return safe_divide(np.diff(acceleration, prepend=acceleration[0]), dt)

def rolling_rms(series, window):
    return np.sqrt(series.pow(2).rolling(window, min_periods=1).mean())