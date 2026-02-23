# src/unit_tests.py
#
# unit tests for drone sim
#
# *** TO RUN: pytest -q ***
#
# Final testing suite
# Partially Gen-AI assisted
import importlib
import os
import sys
import numpy as np
import pytest

# ensure both project root and src directory are importable
# fix for PyCharm test run issues
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
for p in (ROOT_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import kinematics
import simulation
import config

# verifies noise is deterministic and reproducible
def test_target_noisy_maneuver_deterministic():
    t = 1.23
    r_t = np.array([100.0, 200.0, -50.0])
    v_t = np.array([0.0, 0.0, 0.0])
    cfg = {
        'type': 'noisy',
        'base_direction': 'up',  # avoid dependency on TARGET initial velocity
        'noise_strength': 0.25,
        'use_lateral_only': True,
        'seed': 1234, # fixed seed set
        'freqs_hz': [0.07, 0.05, 0.09]
    }

    # calls function used to determine noise generation with fixed ssed (1234)
    u1 = kinematics.target_noisy_maneuver_direction(t, r_t, v_t, cfg)
    u2 = kinematics.target_noisy_maneuver_direction(t, r_t, v_t, cfg)

    #checks outputs are identical
    assert np.allclose(u1, u2)
    assert np.isclose(np.linalg.norm(u1), 1.0, rtol=1e-7, atol=1e-9)

#  test if gravity functions correctly on its own simulation
def test_equations_of_motion_gravity_only(monkeypatch):
    # configure globals inside kinematics to set thrust and drag to 0, not applicable
    monkeypatch.setattr(kinematics, 'G_ACCEL', 9.81, raising=False)
    monkeypatch.setattr(kinematics, 'AIR_DENSITY', 1.225, raising=False)
    monkeypatch.setattr(kinematics, 'GUIDANCE_LAW', 'pp', raising=False)
    monkeypatch.setattr(kinematics, 'PP_CFG', {'damping': 0.0}, raising=False)

    interceptor = {
        'mass': 50.0,
        'Cd': 0.0,
        'A': 0.0,
        'Isp': 0.0,
        'tau': 0.5,
        'fuel': 0.0,
        'burn_rate': 0.0,
    }
    target = {
        'mass': 20.0,
        'Cd': 0.0,
        'A': 0.0,
        'Isp': 0.0,
        'tau': 0.5,
        'fuel': 0.0,
        'burn_rate': 0.0,
        'position': [0.0, 0.0, 0.0],
        'velocity': [0.0, 0.0, 0.0],
        'maneuver': {'type': 'static'},
    }
    # 0 fuel = no thrust

    monkeypatch.setattr(kinematics, 'INTERCEPTOR', interceptor, raising=False)
    monkeypatch.setattr(kinematics, 'TARGET', target, raising=False)

    state = np.zeros(14, dtype=float)

    deriv = kinematics.equations_of_motion(0.0, state)

    assert np.allclose(deriv[0:3], np.zeros(3))
    assert np.allclose(deriv[7:10], np.zeros(3))

    # check how objects respond to gravity with thrust and drag disabled
    gvec = np.array([0.0, -kinematics.G_ACCEL, 0.0])
    assert np.allclose(deriv[3:6], gvec)
    assert np.allclose(deriv[10:13], np.zeros(3))

    # Fuel rates are zero when no burn
    assert deriv[6] == 0.0
    assert deriv[13] == 0.0

# rk4 integrator gravity test - no drag, no thrust
# so we can easily determine where interceptor should be
def test_integrator_gravity_only(monkeypatch):
    import kinematics
    from simulation import Simulation

    # everything except gravity to 0
    monkeypatch.setattr(kinematics, "AIR_DENSITY", 0.0, raising=False)
    monkeypatch.setattr(kinematics, "G_ACCEL", 9.81, raising=False)
    # disable guidance
    monkeypatch.setattr(kinematics, "GUIDANCE_LAW", "pp", raising=False)

    # interceptor and target all set to 0 (apart from mass) to cause no drag
    interceptor = {
        "mass": 100.0, "Cd": 0.0, "A": 0.0,
        "fuel": 0.0, "burn_rate": 0.0,
        "thrust": 0.0, "Isp": 0.0,
        "launch_thrust_frac": 0.0, "tau": 0.5
    }
    target = {
        "mass": 100.0, "Cd": 0.0, "A": 0.0,
        "fuel": 0.0, "burn_rate": 0.0,
        "thrust": 0.0, "Isp": 0.0, "tau": 0.5
    }
    # replace actual variables in kinematics
    monkeypatch.setattr(kinematics, "INTERCEPTOR", interceptor, raising=False)
    monkeypatch.setattr(kinematics, "TARGET", target, raising=False)

    y0 = 1000.0
    interceptor_initial = {"position": [0.0, y0, 0.0], "velocity": [0.0, 0.0, 0.0]}
    target_initial      = {"position": [100.0, y0, 0.0], "velocity": [0.0, 0.0, 0.0]}
    sim = simulation.Simulation(interceptor_initial, target_initial)

    t_end = 0.5  # short run
    t_eval = np.linspace(0.0, t_end, 51)
    res = sim.run(t_span=(0.0, t_end), t_eval=t_eval)

    y_rk4  = res.y[1, -1]
    vvertical_rk4 = res.y[4, -1]

    g = 9.81
    y_true  = y0 - 0.5 * g * t_end**2
    vvertical_true = -g * t_end

    assert np.isclose(y_rk4,  y_true,  rtol=0, atol=3e-3)
    assert np.isclose(vvertical_rk4, vvertical_true, rtol=0, atol=3e-3)

def test_actuator_limit_lag_and_turn_rate(monkeypatch):
    import numpy as np
    import kinematics

    kinematics._DIR_FILTER = {"last_t": None, "u": None}

    # allow changes to config
    cfg = dict(getattr(kinematics, "APN_CFG", {})) or dict(getattr(kinematics, "PN_CFG", {}))
    
    # actuator configuration
    max_turn_rate_deg = 95.0
    monkeypatch.setattr('config.INTERCEPTOR_ACTUATION', {
        "tau_dir": 0.2,
        "max_turn_rate_deg": max_turn_rate_deg
    }, raising=False)


    # start towards +x
    u_start = np.array([1., 0., 0.])
    t_start = 0.0
    u_result = kinematics._apply_actuation_limits(u_start, t_start)

    # command toward -x (180-degree turn)
    u_cmd = np.array([-1., 0., 0.])
    dt = 0.02
    t_next = t_start + dt
    u_next = kinematics._apply_actuation_limits(u_cmd, t_next)

    # check for unit vector
    assert np.isclose(np.linalg.norm(u_next), 1.0, atol=1e-9)

    # check for max allowed angle
    max_rad = np.deg2rad(max_turn_rate_deg) * dt
    dot = float(np.clip(np.dot(u_start, u_next), -1.0, 1.0))
    ang = float(np.arccos(dot))
    assert ang <= max_rad + 1e-6

# after closest approach tests for distance increasing for ≥ 3 seconds being away failure
def test_detect_away_timeout():
    import numpy as np
    from main import detect_away_timeout
    t = np.linspace(0, 10, 101)

    # distance goes down to then increases for 3 seconds
    d = np.concatenate([np.linspace(50, 10, 31), np.linspace(10.1, 30, 70)])
    ok, t_trig = detect_away_timeout(t, d, timeout_sec=3.0)
    assert ok and 5.9 <= t_trig <= 6.1

    # non-constant increase should reset timer so no failure
    d2 = d.copy()
    d2[56] = d2[55] - 0.1  # reset timer
    d2[81] = d2[80] - 0.1  # reset timer again
    ok2, _ = detect_away_timeout(t, d2, timeout_sec=3.0)
    assert not ok2

# tests floor hit termination condition
def test_floor_hit_termination(monkeypatch):
    import numpy as np
    from simulation import Simulation
    import config

    # target starts below ground with downward velocity
    ti = dict(config.TARGET_INITIAL)
    ti['position'] = [0., -10., 0.]
    ti['velocity'] = [0., -50., 0.]

    # interceptor also starts below ground with downward velocity
    ii = dict(config.INTERCEPTOR_INITIAL)
    ii['position'] = [0., -10., 0.]
    ii['velocity'] = [0., -20., 0.]

    sim = Simulation(ii, ti)
    # Run for longer than FLOOR_GRACE_SEC (3.0 seconds) to trigger floor hit
    sol = sim.run(t_span=(0.0, 5.0), t_eval=np.linspace(0.0, 5.0, 26))
    assert getattr(sol, 'floor_miss', False) is True

# test whole run repeatability, same seed = same interaction
def test_whole_run_repeatability(monkeypatch):
    import kinematics
    from simulation import Simulation

    # clear anything left in noise cache
    if hasattr(kinematics, "_NOISE_PARAMS_CACHE"):
        kinematics._NOISE_PARAMS_CACHE.clear()

    # use config conditions to test real repeatability
    from config import INTERCEPTOR_INITIAL, TARGET_INITIAL
    sim1 = simulation.Simulation(INTERCEPTOR_INITIAL, TARGET_INITIAL)
    sim2 = simulation.Simulation(INTERCEPTOR_INITIAL, TARGET_INITIAL)

    t_end = 1.0
    t_eval = np.linspace(0.0, t_end, 101)

    result1 = sim1.run(t_span=(0.0, t_end), t_eval=t_eval)
    # clear cache again before the second run to ensure identical seeded noise
    if hasattr(kinematics, "_NOISE_PARAMS_CACHE"):
        kinematics._NOISE_PARAMS_CACHE.clear()
    result2 = sim2.run(t_span=(0.0, t_end), t_eval=t_eval)

    # Exact equality expected (deterministic RK4 + seeded maneuver noise)
    assert np.array_equal(result1.t, result2.t)
    assert np.array_equal(result1.y, result2.y)

# tests to see how long a single average run takes
def test_performance():
    import time
    import numpy as np
    from simulation import Simulation
    import config
    t0 = time.time()
    sim = Simulation(config.INTERCEPTOR_INITIAL, config.TARGET_INITIAL)
    _ = sim.run(t_span=(0.0, 5.0), t_eval=np.linspace(0.0, 5.0, 251))
    dt = time.time() - t0
    assert dt < 3.0, f"runs too slow: {dt:.2f}s on this hardware"

# --- PP ---

# check if PP works when target x=2, y=0, z=0, guidance law should command 1,0,0
def test_pure_pursuit_guidance_basic():
    r_i = np.array([0.0, 0.0, 0.0])
    v_i = np.array([0.0, 0.0, 0.0])
    r_t = np.array([2.0, 0.0, 0.0])
    v_t = np.array([0.0, 0.0, 0.0])

    u = kinematics.pure_pursuit_guidance(r_i, v_i, r_t, v_t)
    assert np.allclose(u, np.array([1.0, 0.0, 0.0]))

# checks if PP handles edge case where target and interceptor are in the same location
def test_pure_pursuit_guidance_zero_sep():
    r_i = np.array([1.0, 2.0, 3.0])
    v_i = np.array([0.0, 0.0, 0.0])
    r_t = np.array([1.0, 2.0, 3.0])
    v_t = np.array([0.0, 0.0, 0.0])

    u = kinematics.pure_pursuit_guidance(r_i, v_i, r_t, v_t)
    assert np.allclose(u, np.zeros(3))

# --- PN ---

# PN: lateral direction and gain scaling test
# verifies that APN produces lateral element aligned with LOS rotation - increasing N reduces LS component

def test_pn_lateral_direction_and_scaling(monkeypatch):
    r_i = np.array([0., 0., 0.]); v_i = np.array([0., 0., 0.])
    r_t = np.array([100., 0., 0.]); v_t = np.array([-1., 10., 0.]) # fails when a (v_t x [a, b, c]) is 0 as Vc is 0t herefore k_base is 0, note for later

    # baseline PN test
    monkeypatch.setattr(config, 'PN',
        {"N": 4.0, "blend_pp": 0.0, "forward_bias": 0.0, "lat_gain": 1.0}, raising=False)
    importlib.reload(kinematics)
    u1 = kinematics.proportional_navigation_guidance(r_i, v_i, r_t, v_t)

    # stronger (higher N) PN test
    monkeypatch.setattr(config, 'PN',
        {"N": 5.0, "blend_pp": 0.0, "forward_bias": 0.0, "lat_gain": 1.0}, raising=False)
    importlib.reload(kinematics)
    u2 = kinematics.proportional_navigation_guidance(r_i, v_i, r_t, v_t)

    # unit vectors and reduced dot with LOS as N increases
    l = (r_t - r_i) / np.linalg.norm(r_t - r_i)
    assert np.isclose(np.linalg.norm(u1), 1.0, atol=1e-9)
    assert np.isclose(np.linalg.norm(u2), 1.0, atol=1e-9)
    assert np.dot(u2, l) < np.dot(u1, l)

# --- APN ---

# testing APN gates to make sure it only augments when it should
def test_apn_gating_and_augmentation_direction(monkeypatch):
    import numpy as np, kinematics

    # set initial config high gates
    monkeypatch.setattr(kinematics, 'APN_CFG', {
        "N": 4.0, "blend_pp": 0.0, "damping": 0.0, "forward_bias": 0.0, "lat_gain": 1.0,
        "aug_gain": 0.5, "acc_tau": 0.1, "gate_acc": 5.0, "gate_vc": 50.0,
        "a_ref": 1.0, "v_ref": 1.0
    }, raising=False)

    r_i = np.array([0., 0., 0.])
    v_i = np.zeros(3)
    r_t = np.array([100., 0., 0.])
    v_t = np.zeros(3)

    u_pn = kinematics.proportional_navigation_guidance(r_i, v_i, r_t, v_t)
    u_apn = kinematics.augmented_proportional_navigation_guidance(0.0, r_i, v_i, r_t, v_t)
    assert np.allclose(u_apn, u_pn, atol=1e-9)  # should be gated off

    # disable gates and reset filter state
    cfg = dict(kinematics.APN_CFG);
    cfg.update({"gate_acc": 0.0, "gate_vc": 0.0})
    monkeypatch.setattr(kinematics, 'APN_CFG', cfg, raising=False)

    # reset filter state
    if hasattr(kinematics, "_APN_FILTER_STATE"):
        kinematics._APN_FILTER_STATE["last_t"] = None
        kinematics._APN_FILTER_STATE["v_t_f"] = None
        kinematics._APN_FILTER_STATE["a_t_f"] = None

# create scenario with closing velocity and lateral acceleration
    r_i = np.array([0., 0., 0.])
    v_i = np.zeros(3)
    r_t = np.array([100., 0., 0.])

    # First call: target moving toward interceptor at -2 m/s in x direction
    kinematics.augmented_proportional_navigation_guidance(
        0.0, r_i, v_i, r_t, np.array([-2., 0., 0.]))

    # target still closing but now also accelerating upward
    u_apn2 = kinematics.augmented_proportional_navigation_guidance(
        0.1, r_i, v_i, r_t, np.array([-2., 5., 0.]))

    # acceleration estimate should now be ~+50 upwards (y)
    # normal acceleration component should be ~+50 upwards (y) (perpendicular to LOS 1,0,0)
    # should produce APN commanded augmentation in +y direction

    assert not np.allclose(u_apn2, u_pn)
    assert np.isclose(np.linalg.norm(u_apn2), 1.0, atol=1e-9)

    # APN result should have positive y
    assert u_apn2[1] > u_pn[1]
