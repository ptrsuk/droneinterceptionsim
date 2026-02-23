# src/kinematics.py

import numpy as np
from config import G_ACCEL, AIR_DENSITY, INTERCEPTOR, TARGET, GUIDANCE_LAW, PP as PP_CFG, PN as PN_CFG, APN as APN_CFG

# cache for target maneuver noise parameters to avoid recomputing RNG each derivative evaluation
_NOISE_PARAMS_CACHE = {}

def pure_pursuit_guidance(r_i, v_i, r_t, v_t):
    # PP (Pure Pursuit): thrust direction = unit line-of-sight from interceptor to target
    los_vector = r_t - r_i
    dist_to_target = np.linalg.norm(los_vector)
    if dist_to_target > 1e-9:
        return los_vector / dist_to_target
    else:
        return np.zeros(3)


def proportional_navigation_guidance(r_i, v_i, r_t, v_t):
    # PN (Proportional Navigation): thrust direction = unit line-of-sight from interceptor to target + lateral PN
    r = r_t - r_i
    dist = np.linalg.norm(r)
    if dist < 1e-9:
        return np.zeros(3)

    l = r / dist
    v_rel = v_t - v_i

    # closing speed (pos when closing)
    Vc = -float(np.dot(v_rel, l))
    if Vc < 0.0:
        Vc = 0.0

    # LOS rate and lateral PN direction
    lambda_dot = np.cross(l, v_rel) / max(dist, 1e-9)
    n_lat = np.cross(lambda_dot, l) # lateral normal vec - direction perpendiular to LOS and rotation of LOS (most efficient thrust dir)
    nrm = np.linalg.norm(n_lat)
    if nrm > 1e-12:
        n_lat /= nrm
    else:
        return l

    try:
        N = float(PN_CFG.get("N", 3.0))
    except Exception:
        N = 3.0
    # forward bias to preserve closure rate
    try:
        forward_bias = float(PN_CFG.get("forward_bias", 0.0))
    except Exception:
        forward_bias = 0.0
    forward_bias = float(np.clip(forward_bias, 0.0, 5.0))
    # Blend toward PP if requested (1.0 = pure LOS, 0.0 = maximum PN lateral influence)
    try:
        blend_pp = float(PN_CFG.get("blend_pp", 0.0))
    except Exception:
        blend_pp = 0.0
    blend_pp = float(np.clip(blend_pp, 0.0, 1.0))

    try:
        lateral_gain = float(PN_CFG.get("lat_gain", 1.0))
    except Exception:
        lateral_gain = 1.0
    lateral_gain = float(np.clip(lateral_gain, 0.0, 3.0))

    # scaling factor tuned for PN
    k_base = (N * Vc) * 0.02
    k = lateral_gain * (1.0 - blend_pp) * k_base

    #  thrust direction: forward LOS plus lateral PN
    # add forward bias to increase closure rate
    u_vec = (1.0 + forward_bias) * l + k * n_lat
    u_norm = np.linalg.norm(u_vec)
    if u_norm > 1e-9:
        return u_vec / u_norm
    return l


guidance_laws = {
    "pp": pure_pursuit_guidance,
    "pn": proportional_navigation_guidance,
}

# APN
# APN adds target acceleration estimate to PN
# filter is used to estimate target acceleration from velocity

_APN_FILTER_STATE = {
    "last_t": None,
    "v_t_f": None,   # filtered target velocity
    "a_t_f": None,   # filtered target acceleration estimate
}

_DIR_FILTER = {"last_t": None, "u": None}


# interceptor actuator limits (first-order + rate limit)
def _apply_actuation_limits(u_cmd, t):
    from config import INTERCEPTOR_ACTUATION
    tau = float(INTERCEPTOR_ACTUATION.get("tau_dir", 0.12))
    max_deg = float(INTERCEPTOR_ACTUATION.get("max_turn_rate_deg", 120.0))
    max_rate = np.deg2rad(max_deg)  # rad/s

    st = _DIR_FILTER
    # initialize or reset (new sim)
    if st["last_t"] is None or (t is not None and t < (st["last_t"] or 0.0)):
        st["u"] = np.array(u_cmd, dtype=float)
        st["last_t"] = float(t if t is not None else 0.0)
        return st["u"]

    dt = max(1e-6, float(t - st["last_t"]))
    u_cur = st["u"]; u_des = np.array(u_cmd, dtype=float)

    # make sure this is a unit vector!
    u_cur /= max(1e-9, np.linalg.norm(u_cur))
    u_des /= max(1e-9, np.linalg.norm(u_des))

    # first-order move toward command
    alpha = 1.0 - np.exp(-dt / max(1e-6, tau))
    u_tmp = (1.0 - alpha) * u_cur + alpha * u_des
    u_tmp /= max(1e-9, np.linalg.norm(u_tmp))

    # apply max rotation rate
    dot = float(np.clip(np.dot(u_cur, u_tmp), -1.0, 1.0))
    ang = float(np.arccos(dot))
    max_ang = max_rate * dt
    if ang > max_ang:
        # rotate u_cur toward u_tmp by max_ang on the great circle
        axis = np.cross(u_cur, u_tmp)
        n = np.linalg.norm(axis)
        if n < 1e-12:
            u_new = u_cur
        else:
            axis /= n
            # rotate u_cur toward u_tmp by max_ang
            c, s = np.cos(max_ang), np.sin(max_ang)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            u_new = (u_cur * c) + (K @ u_cur) * s + axis * (np.dot(axis, u_cur)) * (1 - c)
    else:
        u_new = u_tmp

    u_new /= max(1e-9, np.linalg.norm(u_new))
    st["u"] = u_new
    st["last_t"] = float(t)
    return u_new

# update filtered estimate of target acceleration

def _apn_update_filters(t, v_t, acc_tau):
    st = _APN_FILTER_STATE
    alpha = 0.0
    if acc_tau > 1e-6:
        # convert continuous tau to discrete alpha; assume modest dt variation
        if st["last_t"] is None:
            alpha = 1.0
        else:
            dt = max(1e-6, float(t - st["last_t"]))
            alpha = 1.0 - np.exp(-dt / float(acc_tau))
    else:
        alpha = 1.0

    if st["v_t_f"] is None:
        st["v_t_f"] = np.array(v_t, dtype=float)
        st["a_t_f"] = np.zeros(3, dtype=float)
        st["last_t"] = float(t)
        return st["a_t_f"]

    # low-pass velocity, then differentiate filtered velocity for accel
    v_prev = st["v_t_f"]
    v_new = (1.0 - alpha) * v_prev + alpha * np.array(v_t, dtype=float)
    dt = max(1e-6, float(t - st["last_t"]))
    a_est = (v_new - v_prev) / dt
    a_f = (1.0 - alpha) * st["a_t_f"] + alpha * a_est
    st["v_t_f"], st["a_t_f"], st["last_t"] = v_new, a_f, float(t)
    return a_f

def augmented_proportional_navigation_guidance(t, r_i, v_i, r_t, v_t):
    # apn = pn + filteresd target accel estimate
    # base PN direction
    u_pn = proportional_navigation_guidance(r_i, v_i, r_t, v_t)

    # estimate target acceleration (filtered)
    acc_tau = float(APN_CFG.get("acc_tau", 0.4))
    a_t = _apn_update_filters(t, v_t, acc_tau)

    # build normal component of target accel relative to LOS
    r = r_t - r_i
    dist = np.linalg.norm(r)
    if dist < 1e-9:
        return u_pn
    l = r / dist
    a_normal = a_t - np.dot(a_t, l) * l
    nrm = np.linalg.norm(a_normal)
    if nrm < 1e-9:
        return u_pn
    a_dir = a_normal / nrm

    # gating to prevent APN from harming PN
    gate_acc = float(APN_CFG.get("gate_acc", 0.5))  # m/s^2 threshold on |a_normal|
    gate_vc  = float(APN_CFG.get("gate_vc", 5.0))   # m/s threshold on closing speed
    Vc = -np.dot((v_t - v_i), l)                    # positive when closing
    if (nrm < gate_acc) or (Vc < gate_vc):
        return u_pn


    # scale augmentation with aug_gain and lat_gain to keep comparable to PN
    aug_gain = float(APN_CFG.get("aug_gain", 0.6))
    aug_gain = max(0.0, min(aug_gain, 2.0))
    lat_gain = float(APN_CFG.get("lat_gain", 0.6))
    lat_gain = max(0.0, min(lat_gain, 3.0))
    a_ref = float(APN_CFG.get("a_ref", 2.0))   # accel scale (m/s^2)
    v_ref = float(APN_CFG.get("v_ref", 10.0))  # speed scale (m/s)

    scale_acc = min(1.0, nrm / max(1e-6, a_ref))
    scale_v   = (Vc / (Vc + v_ref)) if Vc > 0 else 0.0
    k_aug = aug_gain * lat_gain * scale_acc * scale_v

    u_vec = u_pn + k_aug * a_dir
    n = np.linalg.norm(u_vec)
    if n > 1e-9:
        return u_vec / n
    return u_pn

# extend selector
guidance_laws["apn"] = augmented_proportional_navigation_guidance

def target_noisy_maneuver_direction(t, r_t, v_t, cfg):
    seed = int(cfg.get("seed", 0))
    noise_strength = float(cfg.get("noise_strength", 0.2))
    lateral_only = bool(cfg.get("use_lateral_only", True))

    # use provided frequencies or derive/cached from seed
    freqs_in = cfg.get("freqs_hz", None)
    key = None
    if freqs_in is None or len(freqs_in) != 3:
        base = float(cfg.get("freq_base_hz", 0.1))
        jitter = float(cfg.get("freq_jitter_hz", 0.07))
        key = (seed, "auto", base, jitter)
        cached = _NOISE_PARAMS_CACHE.get(key)
        if cached is None:
            rng = np.random.RandomState(seed)
            freqs = [
                max(0.01, base + jitter * (2 * rng.rand() - 1)),
                max(0.01, base + jitter * (2 * rng.rand() - 1)),
                max(0.01, base + jitter * (2 * rng.rand() - 1)),
            ]
            phases = [2 * np.pi * rng.rand(), 2 * np.pi * rng.rand(), 2 * np.pi * rng.rand()]
            _NOISE_PARAMS_CACHE[key] = (freqs, phases)
        else:
            freqs, phases = cached
    else:
        # fixed phases derived from seed to keep deterministic; cache for efficiency
        freqs = list(freqs_in)
        key = (seed, tuple(freqs))
        cached = _NOISE_PARAMS_CACHE.get(key)
        if cached is None:
            rng = np.random.RandomState(seed)
            phases = [2 * np.pi * rng.rand(), 2 * np.pi * rng.rand(), 2 * np.pi * rng.rand()]
            _NOISE_PARAMS_CACHE[key] = (freqs, phases)
        else:
            freqs, phases = cached

    # build noise vector
    n = np.array([
        np.sin(2 * np.pi * freqs[0] * t + phases[0]),
        np.sin(2 * np.pi * freqs[1] * t + phases[1]),
        np.sin(2 * np.pi * freqs[2] * t + phases[2]),
    ], dtype=float)

    # thrust reference direction selection process
    base_choice = str(cfg.get("base_direction", "velocity")).lower()
    if base_choice == "up":
        u_base = np.array([0.0, 1.0, 0.0], dtype=float)
    elif base_choice == "initial_velocity":
        init_v = np.array(TARGET.get("velocity", [1.0, 0.0, 0.0]), dtype=float)
        if np.linalg.norm(init_v) > 1e-6:
            u_base = init_v / np.linalg.norm(init_v)
        else:
            u_base = np.array([0.0, 1.0, 0.0], dtype=float)
    elif base_choice == "custom":
        base_vec = np.array(cfg.get("base_vector", [0.0, 1.0, 0.0]), dtype=float)
        if np.linalg.norm(base_vec) > 1e-6:
            u_base = base_vec / np.linalg.norm(base_vec)
        else:
            u_base = np.array([0.0, 1.0, 0.0], dtype=float)
    else:  # "velocity" (default)
        speed_t = np.linalg.norm(v_t)
        if speed_t > 1e-6:
            u_base = v_t / speed_t
        else:
            init_v = np.array(TARGET.get("velocity", [1.0, 0.0, 0.0]), dtype=float)
            if np.linalg.norm(init_v) > 1e-6:
                u_base = init_v / np.linalg.norm(init_v)
            else:
                u_base = np.array([0.0, 1.0, 0.0], dtype=float)

    # noise related to base direction option
    if lateral_only:
        n = n - np.dot(n, u_base) * u_base
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-6:
            # pick any vector perpendicular to u_base deterministically
            ref = np.array([0.0, 1.0, 0.0]) if abs(u_base[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            n = np.cross(u_base, ref)
            n /= np.linalg.norm(n)
        else:
            n /= n_norm
    else:
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-6:
            n /= n_norm

    # combine base velocity and noise and normalise
    eps = np.clip(noise_strength, 0.0, 1.0)
    u = (1.0 - eps) * u_base + eps * n
    u_norm = np.linalg.norm(u)
    if u_norm > 1e-6:
        return u / u_norm
    else:
        return u_base

def equations_of_motion(t, state):
# interceptor vs target, with gravity, drag, thrust, and fuel
# target state:

    # unpack interceptor
    r_i = state[0:3] # position vector [xi, yi, zi]
    v_i = state[3:6] # velocity vector [vxi, vyi, vzi]
    fuel_i = state[6] # scalar fuel value
    # unpack target
    r_t = state[7:10] # position vector [xt, yt, zt]
    v_t = state[10:13] # velocity vector [vxt, vyt, vzt]
    fuel_t = state[13] # scalar fuel value

    # *** interceptor physics ***
    accel_gravity = np.array([0.0, -G_ACCEL, 0.0])
    speed_i = np.linalg.norm(v_i)
    if speed_i > 0:
        drag_mag_i = 0.5 * AIR_DENSITY * INTERCEPTOR["Cd"] * INTERCEPTOR["A"] * speed_i**2
        a_drag_i = -(drag_mag_i / INTERCEPTOR["mass"]) * (v_i / speed_i)
    else:
        a_drag_i = np.zeros(3)

    # formulate guidance direction (updated every evaluation)
    guidance_func = guidance_laws.get(GUIDANCE_LAW, pure_pursuit_guidance)
    if GUIDANCE_LAW == "apn":
        u_cmd = augmented_proportional_navigation_guidance(t, r_i, v_i, r_t, v_t)
    else:
        u_cmd = guidance_func(r_i, v_i, r_t, v_t)

    u_guidance = u_cmd
    # guidance law specific damping
    try:
        if GUIDANCE_LAW == "pn":
            k_damp = float(PN_CFG.get("damping", 0.0))
        elif GUIDANCE_LAW == "apn":
            k_damp = float(APN_CFG.get("damping", 0.0))
        else: # pp
            k_damp = float(PP_CFG.get("damping", 0.0))
    except Exception:
        k_damp = 0.0

    # lateral velocity damping for both guidance modes
    if k_damp > 0.0 and speed_i > 1e-6:
        v_parallel = np.dot(v_i, u_cmd) * u_cmd
        v_perp = v_i - v_parallel
        nudging = -k_damp * v_perp / max(speed_i, 1e-6)
        u_tmp = u_cmd + nudging
        norm_u = np.linalg.norm(u_tmp)
        if norm_u > 1e-6:
            u_guidance = u_tmp / norm_u
        else:
            u_guidance = u_cmd

    if GUIDANCE_LAW == "pp":
        g_comp_gain = float(PP_CFG.get("gravity_comp", 0.0))
        if g_comp_gain > 0.0:
            # Calculate current max thrust acceleration to determine compensation amount
            exhaust_vel_i = INTERCEPTOR["Isp"] * G_ACCEL
            f_max_i = INTERCEPTOR["burn_rate"] * exhaust_vel_i
            tau_i = INTERCEPTOR["tau"]
            thrust_i = f_max_i * (1 - np.exp(-t / tau_i))
            accel_thrust_mag = thrust_i / INTERCEPTOR["mass"]

            if accel_thrust_mag > 1e-6:
                # Add a vertical component to the guidance vector to counteract gravity
                up_vector = np.array([0.0, 1.0, 0.0])
                compensation = up_vector * g_comp_gain * (G_ACCEL / accel_thrust_mag)
                
                # Add to guidance and re-normalize to keep it a unit vector
                u_guidance = u_guidance + compensation
                u_guidance /= np.linalg.norm(u_guidance)

    # no additional floor bias; guidance is used as-is
    u_guidance = _apply_actuation_limits(u_guidance, t)

    # calculate thrust ramp-up for interceptor
    exhaust_vel_i = INTERCEPTOR["Isp"] * G_ACCEL
    f_max_i = INTERCEPTOR["burn_rate"] * exhaust_vel_i
    tau_i = INTERCEPTOR["tau"]
    thrust_i = f_max_i * (1 - np.exp(-t / tau_i))

    # thrust and fuel logic for interceptor
    if fuel_i > 0:
        a_thrust_i = (thrust_i / INTERCEPTOR["mass"]) * u_guidance
        fuel_i_dot = -INTERCEPTOR["burn_rate"]
    else:
        a_thrust_i = np.zeros(3)
        fuel_i_dot = 0.0

    a_i = accel_gravity + a_drag_i + a_thrust_i # total acceleration of the interceptor (after gravity, drag, and thrust)

    # calculate thrust ramp-up for target
    speed_t = np.linalg.norm(v_t)
    if speed_t > 0:
        drag_mag_t = 0.5 * AIR_DENSITY * TARGET["Cd"] * TARGET["A"] * speed_t**2
        a_drag_t = -(drag_mag_t / TARGET["mass"]) * (v_t / speed_t)
    else:
        a_drag_t = np.zeros(3)
    # calculate thrust ramp-up for target
    exhaust_vel_t = TARGET["Isp"] * G_ACCEL # exhaust_vel_i = exhaust velocity of interceptor
    f_max_t = TARGET["burn_rate"] * exhaust_vel_t # f_max_i = maximum possible thrust =burn rate * exhaust vel
    tau_t = TARGET["tau"] # time to ramp up to max thrust
    thrust_t = f_max_t * (1 - np.exp(-t / tau_t)) # interceptor actual thrust

    # thrust and fuel logic for target
    maneuver_cfg = TARGET.get("maneuver", {})
    maneuver_type = maneuver_cfg.get("type", "none")
    if maneuver_type == "static":
        # temp code for static target
        a_thrust_t = np.zeros(3)
        fuel_t_dot = 0.0
        a_t = np.zeros(3)
    else:
        if fuel_t > 0:
            if maneuver_type == "noisy":
                u_target = target_noisy_maneuver_direction(t, r_t, v_t, maneuver_cfg)
            else:
                u_target = np.zeros(3)
            a_thrust_t = (thrust_t / TARGET["mass"]) * u_target
            fuel_t_dot = -TARGET["burn_rate"]
        else:
            a_thrust_t = np.zeros(3)
            fuel_t_dot = 0.0
        a_t = accel_gravity + a_drag_t + a_thrust_t

    # packing derivatives for return
    dr_i = v_i
    dv_i = a_i
    dr_t = v_t
    dv_t = a_t

    return np.concatenate((dr_i, dv_i, [fuel_i_dot], dr_t, dv_t, [fuel_t_dot]))