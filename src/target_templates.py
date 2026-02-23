# src/target_templates.py

# target templates
# templates have a "maneuver" section which adds noise to thrust dynamics
# maneuver options:
#   type:
#   "none": no additional settings
#   "noise": noise added to movement/thrust vectoring
#     OPTIONS:
#     - noise_strength: blend between direction and noise inputs - 0=straight 1=full noise
#     - use_lateral_only: True = noise is linked to velocity
#     - seed: seed-based noise generation to keep consistent noise between runs of each algorithm
#     - freqs_hz: [fx, fy, fz] - set fixed frequencies

TARGET_TEMPLATES = {

    # 0 - completely static target for testing
    "static_zero": {
        "mass": 20.0,
        "Cd": 0.0,
        "A": 0.0,
        "Isp": 0.0,
        "tau": 1.0,
        "fuel": 0.0,
        "burn_rate": 0.0,
        "position": [8000, 5000, 8000],
        "velocity": [0.0, 0.0, 0.0],
        "maneuver": {"type": "static"}
    },

    #
    # 1 - drone hover with very slight movement
    # near hover with very low (2%) noise
    # no initial motion
    #

    "drone_hover": {
        "mass": 20.0,
        "Cd": 0.3,
        "A": 0.1,
        "Isp": 500.0,
        "tau": 0.3,
        "fuel": 20.0,
        "burn_rate": 0.04,
        "position": [8000, 3000, 6000],
        "velocity": [0.0, 0.0, 0.0],
        "maneuver": {
            "type": "noisy",
            "base_direction": "up",
            "noise_strength": 0.02,
            "use_lateral_only": True,
            "seed": 1001,
            "freqs_hz": [0.05, 0.04, 0.06]
        }
    },

    #
    # 2 - drone hover with slight drift
    # near hover with slightly higher noise
    # slight initial x axis motion
    #

    "drone_hover_drift": {
        "mass": 20.0,
        "Cd": 0.3,
        "A": 0.1,
        "Isp": 500.0,
        "tau": 0.3,
        "fuel": 20.0,
        "burn_rate": 0.04,
        "position": [7000, 4500, 3500],
        "velocity": [2.0, 0.0, 0.0],
        "maneuver": {
            "type": "noisy",
            "base_direction": "up",
            "noise_strength": 0.04,
            "use_lateral_only": True,
            "seed": 1001,
            "freqs_hz": [0.06, 0.05, 0.04]
        }
    },

    #
    # 3 - drone hover with forward glide
    # hover but with some momentum and higher noise
    # more unpredictable than other hovers
    #

    "drone_hover_glide": {
        "mass": 20.0,
        "Cd": 0.3,
        "A": 0.1,
        "Isp": 500.0,
        "tau": 0.3,
        "fuel": 20.0,
        "burn_rate": 0.04445,
        "position": [5000, 4000, 4000],
        "velocity": [4.5, 0.0, 0.0],
        "maneuver": {
            "type": "noisy",
            "base_direction": "up",
            # a bit more noise but still moderate
            "noise_strength": 0.08,
            "use_lateral_only": True,
            "seed": 1001,
            "freqs_hz": [0.07, 0.08, 0.09]
        }
    },

    #
    # 4 - drone dive with slight horizontal movement
    # dives downwards with a small forward drift, straight path
    #

    "drone_dive_light": {
        "mass": 20.0,
        "Cd": 0.3,
        "A": 0.1,
        "Isp": 500.0,
        "tau": 0.3,
        "fuel": 20.0,
        "burn_rate": 0.045,
        "position": [3000, 5000, 6000],
        "velocity": [2.0, -2.0, 0.5],
        "maneuver": {
            "type": "noisy",
            "base_direction": "custom",
            # Thrust unit vector: mostly forward with enough up to nearly balance gravity -> slight descent
            # Approx unit vector after normalization
            "base_vector": [0.66, 0.75, 0.0],
            "noise_strength": 0.03,
            "use_lateral_only": True,
            "seed": 1001,
            "freqs_hz": [0.06, 0.05, 0.07]
        }
    },

    #
    # 5 - drone glide slightly downward with medium horizontal speed
    # steady forward glide with mild descent; moderate noise adds some unpredictability
    #

    "drone_glide_down_medium": {
        "mass": 20.0,
        "Cd": 0.3,
        "A": 0.1,
        "Isp": 500.0,
        "tau": 0.3,
        "fuel": 20.0,
        "burn_rate": 0.05,
        "position": [2000, 3000, 4000],
        "velocity": [8.0, -1.0, 3.0],
        "maneuver": {
            "type": "noisy",
            "base_direction": "custom",
            # Thrust unit vector: forward glide with moderate up -> mild descent under gravity
            # Approx unit vector after normalization
            "base_vector": [0.88, 0.47, 0.08],
            "noise_strength": 0.06,
            "use_lateral_only": True,
            "seed": 1001,
            "freqs_hz": [0.08, 0.06, 0.07]
        }
    },

    #
    # 6 - drone fast horizontal flight
    # higher, realistic horizontal speed for a drone, mostly straight-line with light noise
    #

    "drone_fast_horizontal": {
        "mass": 20.0,
        "Cd": 0.3,
        "A": 0.1,
        "Isp": 500.0,
        "tau": 0.3,
        "fuel": 20.0,
        "burn_rate": 0.06,
        "position": [8000, 2500, 4000],
        "velocity": [25.0, 0.0, 0.0],
        "maneuver": {
            "type": "noisy",
            "base_direction": "custom",
            # Thrust unit vector: nearly level flight; a bit less than hover-up for gentle descent
            # Approx unit vector after normalization
            "base_vector": [0.78, 0.62, 0.0],
            "noise_strength": 0.03,
            "use_lateral_only": True,
            "seed": 1001,
            "freqs_hz": [0.09, 0.07, 0.05]
        }
    },

    #
    # 7 - drone low-level snake
    # low-altitude weave like a snake
    # lateral-only noise at moderate speed
    #

    "drone_low_level_snake": {
        "mass": 20.0,
        "Cd": 0.35,
        "A": 0.12,
        "Isp": 500.0,
        "tau": 0.3,
        "fuel": 20.0,
        "burn_rate": 0.055,
        "position": [1200, 1200, 900],
        "velocity": [15.0, 0.5, 0.0],
        "maneuver": {
            "type": "noisy",
            "base_direction": "custom",
            # Forward with slight up to sustain low-level flight
            "base_vector": [0.98, 0.20, 0.0],
            "noise_strength": 0.10,
            "use_lateral_only": True,
            "seed": 1001,
            "freqs_hz": [0.12, 0.06, 0.18]
        }
    },

    #
    # 8 - drone zigzag climb
    # continuous snake-like climb with wide turns, predictable but aggressive
    #

    "drone_zigzag_climb": {
        "mass": 20.0,
        "Cd": 0.3,
        "A": 0.1,
        "Isp": 500.0,
        "tau": 0.2,
        "fuel": 20.0,
        "burn_rate": 0.16,
        "position": [2500, 1000, 2800],
        "velocity": [28.0, 2.0, 0.5],
        "maneuver": {
            "type": "noisy",
            "base_direction": "custom",
            "base_vector": [0.87, 0.50, 0.0],
            "noise_strength": 0.35,
            "use_lateral_only": True,
            "seed": 1001,
            "freqs_hz": [0.22, 0.17, 0.25]
        }
    },

    #
    # 9 - drone spiral climb
    # upward spiral/orbit, lateral swirl around vertical; harder due to 3D path
    #

    "drone_spiral_climb": {
        "mass": 20.0,
        "Cd": 0.3,
        "A": 0.1,
        "Isp": 500.0,
        "tau": 0.3,
        "fuel": 20.0,
        "burn_rate": 0.052,
        "position": [4000, 1500, 1500],
        "velocity": [6.0, 0.0, 6.0],
        "maneuver": {
            "type": "noisy",
            "base_direction": "up",
            # lateral swirl creates a spiral ascent
            "noise_strength": 0.12,
            "use_lateral_only": True,
            "seed": 1001,
            "freqs_hz": [0.07, 0.05, 0.07]
        }
    },

    #
    # 10 - prop high-speed cruise
    # fixed-wing style fast cruise with gentle banking; harder via speed
    #

    "prop_high_speed_cruise": {
        "mass": 18.0,
        "Cd": 0.22,
        "A": 0.08,
        "Isp": 550.0,
        "tau": 0.4,
        "fuel": 22.0,
        "burn_rate": 0.065,
        "position": [9500, 1500, 600],
        "velocity": [35.0, 0.0, 0.0],
        "maneuver": {
            "type": "noisy",
            "base_direction": "custom",
            # near-level fast cruise with slight nose-up
            "base_vector": [0.96, 0.27, 0.0],
            "noise_strength": 0.05,
            "use_lateral_only": True,
            "seed": 1001,
            "freqs_hz": [0.15, 0.05, 0.12]
        }
    },

    #
    # 11 - prop serpentine descent
    # forward descending S-turns; challenging lateral motion and vertical rate
    #

    "prop_snake_descent": {
        "mass": 18.0,
        "Cd": 0.24,
        "A": 0.08,
        "Isp": 550.0,
        "tau": 0.4,
        "fuel": 22.0,
        "burn_rate": 0.06,
        "position": [7000, 1800, 4000],
        "velocity": [18.0, 0.0, 0.0],
        "maneuver": {
            "type": "noisy",
            "base_direction": "custom",
            # forward with slight down pitch for descent
            "base_vector": [0.95, -0.30, 0.0],
            "noise_strength": 0.14,
            "use_lateral_only": True,
            "seed": 1001,
            "freqs_hz": [0.18, 0.07, 0.16]
        }
    },

    #
    # 12 - quadcopter aggressive rise
    # rapid high-frequency lateral movements at mid altitude; hardest short-range
    #

    "quadcopter_aggressive_rise": {
        "mass": 20.0,
        "Cd": 0.3,
        "A": 0.1,
        "Isp": 500.0,
        "tau": 0.20,
        "fuel": 20.0,
        "burn_rate": 0.12,
        "position": [10000, 2800, 8000],
        "velocity": [30.0, 0.2, 1.0],
        "maneuver": {
            "type": "noisy",
            "base_direction": "up",
            "noise_strength": 0.35,
            "use_lateral_only": True,
            "seed": 1001,
            "freqs_hz": [0.18, 0.16, 0.12]
        }
    }
}