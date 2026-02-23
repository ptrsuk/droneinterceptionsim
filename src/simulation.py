# src/simulation.py
# defines the Simulation class to run the sim using RK4
# was originaly using solve_ivp but it was too slow and not working properly

import numpy as np
from kinematics import equations_of_motion
from config import INTERCEPTOR, TARGET, INTERCEPTION_RADIUS, GUIDANCE_LAW, FLOOR_GRACE_SEC


def interception_event(t, state):
    # triggers when separation distance equals interception radius
    r_i = state[0:3]
    r_t = state[7:10]
    sep = np.linalg.norm(r_t - r_i)
    return sep - INTERCEPTION_RADIUS

# stop when interception is reached
interception_event.terminal = True
interception_event.direction = -1


class Simulation:
    # sets up engagement simulation
    # takes intial states for the interceptor and target
    # interceptor_initial - {'position': [x, y, z], 'velocity': [vx, vy, vz]}
    # target_initial - {'position': [x, y, z], 'velocity': [vx, vy, vz]}

    def __init__(self, interceptor_initial, target_initial):
        # store initial positions and velocities
        self.interceptor_initial_pos = np.array(interceptor_initial['position'], dtype=float)
        self.interceptor_initial_vel = np.array(interceptor_initial['velocity'], dtype=float)
        self.target_initial_pos = np.array(target_initial['position'], dtype=float)
        self.target_initial_vel = np.array(target_initial['velocity'], dtype=float)

    def run(self, t_span, t_eval):
        # create initial state vector: interceptor pos, vel, fuel, then target pos, vel, fuel
        interceptor_state = list(self.interceptor_initial_pos) + list(self.interceptor_initial_vel) + [INTERCEPTOR["fuel"]]
        target_state = list(self.target_initial_pos) + list(self.target_initial_vel) + [TARGET["fuel"]]
        initial_state = np.array(interceptor_state + target_state, dtype=float)

        # use same integrator to ensure comparability
        t0, tf = float(t_span[0]), float(t_span[1])
        t_eval = np.array(t_eval, dtype=float)
        # Ensure t_eval within span and sorted
        t_eval = t_eval[(t_eval >= t0) & (t_eval <= tf)]
        if t_eval.size == 0:
            t_eval = np.array([t0, tf])

        y_dim = initial_state.size
        Y = np.zeros((y_dim, t_eval.size), dtype=float)
        T = np.zeros((t_eval.size,), dtype=float)

        # RK4 stepping
        h_max = 0.02  # 20 ms cap on step - trade off between computational efficiency and fidelity
        state = initial_state.copy()
        t_cur = t0
        # initialize first sample if t_eval starts at t0
        next_idx = 0
        if abs(t_eval[0] - t0) < 1e-12:
            Y[:, 0] = state
            T[0] = t0
            next_idx = 1

        hit_time = None
        sep_prev = None
        floor_miss_time = None
        max_substeps = 200000 # hard cap on substeps
        substeps = 0

        # RK4 advance step
        def rk4_step(t, y, h):
            k1 = equations_of_motion(t, y)
            k2 = equations_of_motion(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = equations_of_motion(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = equations_of_motion(t + h, y + h * k3)
            return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # iterate over evaluation targets
        while next_idx < t_eval.size and substeps < max_substeps:
            t_target = t_eval[next_idx]
            # Integrate from t_cur to t_target
            while t_cur < t_target - 1e-12 and substeps < max_substeps:
                h = min(h_max, t_target - t_cur)
                state = rk4_step(t_cur, state, h)
                t_cur += h
                substeps += 1

                # check for intrception radius crossing
                r_i = state[0:3]
                r_t = state[7:10]
                sep = np.linalg.norm(r_t - r_i)
                if sep_prev is not None and sep <= INTERCEPTION_RADIUS <= sep_prev:
                    hit_time = t_cur
                    break
                sep_prev = sep

                # floor-hit miss condition after grace period
                if t_cur >= float(FLOOR_GRACE_SEC):
                    yi = r_i[1]
                    yt = r_t[1]
                    if yi <= 0.0 or yt <= 0.0:
                        floor_miss_time = t_cur
                        break

            # store sample (either reached target time or event)
            Y[:, next_idx] = state
            T[next_idx] = t_cur
            next_idx += 1

            if hit_time is not None or floor_miss_time is not None:
                break

        # end early
        Y = Y[:, :next_idx]
        T = T[:next_idx]

        # build result object
        # tries to mirror solve_ivp
        class Result:
            pass
        res = Result()
        res.t = T # sim times array
        res.y = Y # sim var states array
        res.success = True
        res.t_events = [np.array([hit_time])] if hit_time is not None else [np.array([])]
        # additional metadata for floor miss case
        res.floor_miss = floor_miss_time is not None
        res.floor_miss_time = floor_miss_time
        return res