
# src/main.py

import os
import matplotlib
# choose backend based on env flags so batch png savve runs well

if os.environ.get("SIM_SAVE_PNG") == "1" or os.environ.get("BATCH_MODE") == "1":
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as mpl
from simulation import Simulation
from config import INTERCEPTOR_INITIAL, TARGET_INITIAL, INTERCEPTION_RADIUS, AWAY_TIMEOUT_SEC, GUIDANCE_LAW, TARGET_TEMPLATE_NAME

def detect_away_timeout(times, separations, timeout_sec, eps=1e-6):
    # find closest point of approach (CPA)
    if len(times) == 0:
        return False, None
    min_idx = int(np.argmin(separations))
    # scan for continuous increasing separation after CPA
    start_idx = None
    for k in range(min_idx + 1, len(times)):
        if separations[k] > separations[k - 1] + eps: # epsilon to detect TRUE increase (fl p)
            if start_idx is None:
                start_idx = k - 1
            # check duration from start_idx to current index
            if times[k] - times[start_idx] >= timeout_sec:
                # Interpolate exact time if needed, but nearest sample is acceptable here
                return True, times[k]
        else:
            # reset if not strictly increasing
            start_idx = None
    return False, None


def main(save_path: str | None = None, title_suffix: str = ""):
    # main function - sets up simulation, runs it & plots results
    try:
        print("~ Simulation setting up ~")
        print(f"~ Active guidance law: {GUIDANCE_LAW} ~")
        interceptor_initial = INTERCEPTOR_INITIAL
        target_initial = TARGET_INITIAL

        # define simulation time based on fuel and burn rate
        from config import INTERCEPTOR, TARGET
        t_start = 0
        burn_time_i = (INTERCEPTOR["fuel"] / INTERCEPTOR["burn_rate"]) if INTERCEPTOR["burn_rate"] > 0 else 0.0
        burn_time_t = (TARGET["fuel"] / TARGET["burn_rate"]) if TARGET["burn_rate"] > 0 else 0.0
        t_end = burn_time_i + 10  # simulate only until interceptor burnout + small buffer
        t_points = np.linspace(t_start, t_end, 200)  # 200 points for a smooth plot

        print("~ Creating simulation object ~")

        # *** run simulation - RK4 in sim class ***

        sim = Simulation(interceptor_initial, target_initial)

        print("~ Running simulation ~")
        solution = sim.run(t_span=(t_start, t_end), t_eval=t_points)

        # --- metrics: fuel usage and acceleration (do not alter kinematics) ---
        try:
            times_full = solution.t
            vel_i_full = solution.y[3:6, :]
            fuel_full = solution.y[6, :]

            # Fuel used (kg)
            fuel_initial = float(fuel_full[0])
            fuel_remaining = float(fuel_full[-1])
            fuel_used_kg = max(0.0, fuel_initial - max(0.0, fuel_remaining))

            # Acceleration magnitude from velocity derivative via time-gradient
            if times_full.size >= 2:
                a_components = np.gradient(vel_i_full, times_full, axis=1)
                accel_mag = np.linalg.norm(a_components, axis=0)
                max_accel_mps2 = float(np.max(accel_mag))
                duration = float(times_full[-1] - times_full[0])
                avg_accel_mps2 = float((np.trapezoid(accel_mag, times_full) / duration) if duration > 0 else 0.0)
            else:
                max_accel_mps2 = 0.0
                avg_accel_mps2 = 0.0
        except Exception:
            fuel_used_kg = float('nan')
            max_accel_mps2 = float('nan')
            avg_accel_mps2 = float('nan')

        # compute separations over time
        diffs = solution.y[7:10, :] - solution.y[0:3, :]
        dists = np.linalg.norm(diffs, axis=0)
        min_idx = int(np.argmin(dists))
        min_sep = float(dists[min_idx])

        # away timeout failure condition
        away_failed, away_time = detect_away_timeout(solution.t, dists, AWAY_TIMEOUT_SEC)

        # floor-hit failure condition
        floor_failed = bool(getattr(solution, 'floor_miss', False))
        floor_time = float(getattr(solution, 'floor_miss_time', np.nan)) if floor_failed else None

        collided = False
        collision_point = None
        collision_time = None
        if hasattr(solution, 't_events') and solution.t_events and len(solution.t_events) > 0 and len(solution.t_events[0]) > 0:
            collided = True
            collision_time = solution.t_events[0][0]

        print(f"Simulation completed successfully - {solution.success}")
        print(f"Overall state variables tracked / time points tracked - {solution.y.shape}")
        print(f"Min separation: {min_sep:.2f} m | Threshold: {INTERCEPTION_RADIUS:.2f} m | AwayTimeout: {AWAY_TIMEOUT_SEC:.1f}s")
        if collided and collision_time is not None:
            print(f"Intercept time: {float(collision_time):.2f} s")

        # *** result processing and visualising ***

        # place trajectories into solution variables
        interceptor_trajectory = solution.y[0:3, :]
        target_trajectory = solution.y[7:10, :]
        times = solution.t

        # apply failure plot truncation priority: floor hit, away timeout, collision
        # cleans up before plotting
        if floor_failed and (collision_time is None or floor_time < collision_time) and (not away_failed or floor_time < away_time):
            idx_cut = (np.abs(times - floor_time)).argmin()
            interceptor_trajectory = interceptor_trajectory[:, :idx_cut+1]
            target_trajectory = target_trajectory[:, :idx_cut+1]
            times = times[:idx_cut+1]
            collided = False
            away_failed = False
        elif away_failed and (collision_time is None or away_time < collision_time):
            idx_cut = (np.abs(times - away_time)).argmin()
            interceptor_trajectory = interceptor_trajectory[:, :idx_cut+1]
            target_trajectory = target_trajectory[:, :idx_cut+1]
            times = times[:idx_cut+1]
            collided = False
        elif collided and collision_time is not None:
            idx_col = (np.abs(times - collision_time)).argmin()
            interceptor_trajectory = interceptor_trajectory[:, :idx_col+1]
            target_trajectory = target_trajectory[:, :idx_col+1]
            times = times[:idx_col+1]

        print(f"Interceptor trajectory ^ / ^ - {interceptor_trajectory.shape}")
        print(f"Target trajectory ^ / ^ - {target_trajectory.shape}")

        # create 3d plot
        print("~ Creating 3D plot ~")
        figure = mpl.figure(figsize=(16, 12))
        ax = figure.add_subplot(111, projection='3d')

        # plot trajectories with mpl (have to swap y and z for correct axis mapping - weird plot library)
        ax.plot(interceptor_trajectory[0], interceptor_trajectory[2], interceptor_trajectory[1],
                label='Interceptor Trajectory', color='b')
        ax.plot(target_trajectory[0], target_trajectory[2], target_trajectory[1], label='Target Trajectory', color='r')

        # add time labels every 10 seconds
        label_interval = 10
        # only label up to the last time
        last_time = solution.t[-1]
        label_times = np.arange(label_interval, last_time, label_interval)
        for label_time in label_times:
            idx = (np.abs(solution.t - label_time)).argmin()
            if idx < interceptor_trajectory.shape[1] and idx < target_trajectory.shape[1]:
                # Interceptor
                ax.text(interceptor_trajectory[0, idx], interceptor_trajectory[2, idx], interceptor_trajectory[1, idx],
                        f"{int(round(solution.t[idx]))}s", color='blue', fontsize=10, ha='center', va='center')
                # Target
                ax.text(target_trajectory[0, idx], target_trajectory[2, idx], target_trajectory[1, idx],
                        f"{int(round(solution.t[idx]))}s", color='red', fontsize=10, ha='center', va='center')

        # mark start and end points (swap y and z)
        ax.scatter(interceptor_initial['position'][0], interceptor_initial['position'][2],
                   interceptor_initial['position'][1],
                   color='blue', marker='o', s=100, label='Interceptor Start')
        ax.scatter(target_initial['position'][0], target_initial['position'][2], target_initial['position'][1],
                   color='red',
                   marker='o', s=100, label='Target Start')
        ax.scatter(interceptor_trajectory[0, -1], interceptor_trajectory[2, -1], interceptor_trajectory[1, -1],
                   color='blue', marker='x', s=100, label='Interceptor End')
        ax.scatter(target_trajectory[0, -1], target_trajectory[2, -1], target_trajectory[1, -1], color='red',
                   marker='x',
                   s=100, label='Target End')

        # set plot labels (x = downrange, y = vertical/altitude, z = crossrange)
        title = f'Engagement Simulation ({GUIDANCE_LAW.upper()})'
        if title_suffix:
            title += f" {title_suffix}"
        #confusing naming due to y / z swap earlier
        ax.set_xlabel('X position (Crossrange, m)')
        ax.set_ylabel('Z position (Downrange, m)')
        ax.set_zlabel('Y position (Altitude, m)')
        ax.set_title(title)
        # collision/away markers and annotation
        if collided and collision_time is not None:
            # derive collision point as midpoint at nearest index
            idx_col = (np.abs(solution.t - collision_time)).argmin()
            r_i_col = solution.y[0:3, idx_col]
            r_t_col = solution.y[7:10, idx_col]
            collision_point = (r_i_col + r_t_col) / 2.0
            ax.scatter(collision_point[0], collision_point[2], collision_point[1],
                       color='green', marker='*', s=200, label='Collision')
            ax.text(collision_point[0], collision_point[2], collision_point[1],
                    f"Hit @ {collision_time:.2f}s\nsep~{min_sep:.1f}m",
                    color='green', fontsize=10, ha='left', va='bottom')
        elif floor_failed and len(times) > 0:
            ax.scatter(interceptor_trajectory[0, -1], interceptor_trajectory[2, -1], interceptor_trajectory[1, -1],
                       color='black', marker='x', s=140, label='Failed (floor contact)')
        elif away_failed and len(times) > 0:
            # mark failure end point
            ax.scatter(interceptor_trajectory[0, -1], interceptor_trajectory[2, -1], interceptor_trajectory[1, -1],
                       color='black', marker='x', s=120, label='Failed (moving away)')

        ax.legend()
        ax.grid(True)

        # text summary in the window
        status_text = (
            f"Collision: {collided} | Min separation: {min_sep:.2f} m | Threshold: {INTERCEPTION_RADIUS:.2f} m"
            + (f" | Failed: floor contact after 3.0s" if floor_failed and not collided else "")
            + (f" | Failed: moving away > {AWAY_TIMEOUT_SEC:.1f}s since CPA" if away_failed and not collided and not floor_failed else "")
        )
        figure.text(0.02, 0.02, status_text, fontsize=12, color=('green' if collided else 'red'))

        # set axis limits based on actual min/max reached by both trajectories
        x_min = min(interceptor_trajectory[0].min(), target_trajectory[0].min())
        x_max = max(interceptor_trajectory[0].max(), target_trajectory[0].max())
        y_min = min(interceptor_trajectory[2].min(), target_trajectory[2].min())
        y_max = max(interceptor_trajectory[2].max(), target_trajectory[2].max())
        z_min = min(interceptor_trajectory[1].min(), target_trajectory[1].min())
        z_max = max(interceptor_trajectory[1].max(), target_trajectory[1].max())
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        if save_path:
            # save for batch mode with identical styling to regular main run
            mpl.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            print("Displaying plot...")
            mpl.show()
            print("Plot displayed successfully")

        # return metrics so batch runner can log results in csv
        result = {
            "success": bool(collided),
            "time_to_hit": (None if not collided else float(collision_time)),
            "min_separation": float(min_sep),
            "fail_type": ("none" if collided else ("floor" if floor_failed else ("away" if away_failed else "none"))),
            "fuel_used_kg": float(fuel_used_kg),
            "max_accel_mps2": float(max_accel_mps2),
            "avg_accel_mps2": float(avg_accel_mps2),
        }
        return result

    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "time_to_hit": None, "min_separation": float('nan'), "fail_type": "error"}


if __name__ == "__main__":
    main()
