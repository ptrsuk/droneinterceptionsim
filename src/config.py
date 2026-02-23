# src/config.py


### --- INTERCEPTOR PROPERTIES --- ###

# interceptor and target positions
INTERCEPTOR_INITIAL = {
    'position': [1500, 40, 2000],      # x (downrange), y (vertical/altitude), z (crossrange)
    'velocity': [0, 1, 0]     # x (vel downrange), y (vel vertical/altitude), z (vel crossrange)
}

# interceptor aero properties
INTERCEPTOR = {
    "mass": 210.0, # kg
    "Cd":   0.42, # drag coefficient
    "A":    0.05, # cross-sectional area (m²)
    "Isp": 250.0,  # specific impulse (s)
    "tau": 0.7,  # engine startup time constant (s)
    "fuel": 190,  # total fuel mass (kg)
    "burn_rate": 6,  # fuel consumption rate (kg/s)
    "launch_thrust_frac": 0.2 # amount of thrust at launch (%)
}

# interceptor actuation dynamics (limits how fast thrust direction can change)
INTERCEPTOR_ACTUATION = {
    "tau_dir": 0.2,            # first-order response (seconds)
    "max_turn_rate_deg": 95.0   # degrees per second cap on direction change rate
}

# target template selection
from target_templates import TARGET_TEMPLATES
TARGET_TEMPLATE_NAME = "drone_zigzag_climb"  # select template from target_templates.py
TARGET = TARGET_TEMPLATES[TARGET_TEMPLATE_NAME]
TARGET_INITIAL = {
    'position': TARGET['position'],
    'velocity': TARGET['velocity']
}

### --- PHYSICS PROPERTIES --- ###

# gravitational acceleration (m/s²)
G_ACCEL = 9.81

# air density at sea level (kg/m³)
AIR_DENSITY = 1.225

GUIDANCE_LAW = "pn"

# Pure Pursuit tuning
PP = {
    "damping": 0.7, # lower for PP to avoid understeer (0 min - 1 max)
    "gravity_comp": 1.05,
}
# PN tuning
PN = {
    # N = 4.0 used to be 3.5 backup value
    "N": 4.0,
    # blend between PN lateral direction and LOS
    # 0.0 = pure PN lateral direction only - 1.0 = pure LOS
    "blend_pp": 0.08,
    "damping": 0.1,
    "forward_bias": 0.4,
    "lat_gain": 0.7
}

# apn tuning
APN = {
    "N": 4.0,
    "blend_pp": 0.08,
    "damping": 0.1,
    "forward_bias": 0.4,
    "lat_gain": 0.7,
    # augmentation strength and accel filter
    "aug_gain": 0.27,    # scale for the target normal-accel term
    "acc_tau": 0.6,       # seconds, low-pass filter acceleration time-constant for accel estimate

    "gate_acc": 1.6,   # m/s^2 # accel gate
    "gate_vc": 6.5,    # m/s  # closing velocity gate
    "a_ref": 1.3,      # m/s^2
    "v_ref": 6.0      # m/s
}

# interception parameters
INTERCEPTION_RADIUS = 15.0  # meters
# miss if moving away longer than this after closest approach
AWAY_TIMEOUT_SEC = 3.0
 # grace period before floor-hit ends the sim
FLOOR_GRACE_SEC = 3.0