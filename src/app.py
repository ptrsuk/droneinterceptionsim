
# src/app.py
# streamlit web ui version of the simulator
# run with: streamlit run src/app.py

import os
import sys
import copy
import importlib
import io

os.environ["SIM_SAVE_PNG"] = "1"
os.environ["BATCH_MODE"] = "1"

# fix imports so it can find the src modules
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))
for p in (ROOT_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mpl

import config as cfg
import target_templates as tt
import kinematics as kine
import simulation as sim_mod
from main import detect_away_timeout

st.set_page_config(page_title="Drone Interception Simulator", layout="wide",
                   initial_sidebar_state="expanded")

# readable names for guidance laws and targets
GUIDANCE_NAMES = {
    "pp": "Pure Pursuit (PP)",
    "pn": "Proportional Navigation (PN)",
    "apn": "Augmented Proportional Navigation (APN)",
}
TARGET_NAMES = {
    "static_zero": "Static Target (Testing)",
    "drone_hover": "Drone Hover",
    "drone_hover_drift": "Drone Hover with Drift",
    "drone_hover_glide": "Drone Hover with Glide",
    "drone_dive_light": "Drone Light Dive",
    "drone_glide_down_medium": "Drone Medium Glide Down",
    "drone_fast_horizontal": "Drone Fast Horizontal",
    "drone_low_level_snake": "Drone Low-Level Snake",
    "drone_zigzag_climb": "Drone Zigzag Climb",
    "drone_spiral_climb": "Drone Spiral Climb",
    "prop_high_speed_cruise": "Prop High Speed Cruise",
    "prop_snake_descent": "Prop Serpentine Descent",
    "quadcopter_aggressive_rise": "Quadcopter Aggressive Rise",
}
GUIDANCE_DESC = {
    "pp": "Steers the interceptor directly toward the target's current position. Simple but tends to fall behind manoeuvring targets.",
    "pn": "Steers proportionally to the rate of change of the line-of-sight angle, leading the target for more efficient interception.",
    "apn": "Extends PN by estimating and compensating for target acceleration, improving performance against manoeuvring targets.",
}


# *** sidebar config ***

st.sidebar.markdown("## Configuration")

guidance_law = st.sidebar.selectbox(
    "Guidance Law", list(GUIDANCE_NAMES.keys()),
    format_func=lambda x: GUIDANCE_NAMES[x], index=1,
)
target_names = list(tt.TARGET_TEMPLATES.keys())
target_name = st.sidebar.selectbox(
    "Target Scenario", target_names,
    format_func=lambda x: TARGET_NAMES.get(x, x),
    index=target_names.index("drone_zigzag_climb"),
)
template = tt.TARGET_TEMPLATES[target_name]
default_seed = int(template.get("maneuver", {}).get("seed", 1001))
seed = st.sidebar.number_input("Noise Seed", min_value=1, max_value=99999, value=default_seed)

st.sidebar.markdown("---")
run_clicked = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)


# *** page header ***

st.markdown(
    "<h1 style='margin-bottom:0'>Drone Interception Simulator</h1>"
    "<p style='color:grey;margin-top:0'>3D physics-based comparative analysis of PP, PN, and APN guidance laws</p>",
    unsafe_allow_html=True,
)
col1, col2 = st.columns([2, 1])
with col1:
    st.info(f"**{GUIDANCE_NAMES[guidance_law]}** — {GUIDANCE_DESC[guidance_law]}")
with col2:
    st.markdown(f"**Target:** {TARGET_NAMES.get(target_name, target_name)}  \n**Seed:** {seed}")


# *** simulation runner - same approach as batch_run_main.py ***
# sets config, reloads modules, runs sim, computes metrics

def run_simulation(guidance_law, target_name, seed):
    cfg.GUIDANCE_LAW = guidance_law
    cfg.TARGET_TEMPLATE_NAME = target_name
    target_cfg = copy.deepcopy(tt.TARGET_TEMPLATES[target_name])
    if isinstance(target_cfg.get("maneuver"), dict) and target_cfg["maneuver"].get("type") == "noisy":
        target_cfg["maneuver"]["seed"] = int(seed)
    cfg.TARGET = target_cfg
    cfg.TARGET_INITIAL = {"position": target_cfg["position"], "velocity": target_cfg["velocity"]}

    # reload modules with constants - same as batch_run_main
    importlib.reload(kine)
    importlib.reload(sim_mod)

    interceptor_initial = dict(cfg.INTERCEPTOR_INITIAL)
    target_initial = dict(cfg.TARGET_INITIAL)

    # sim time setup - same as main.py
    t_start = 0
    burn_time_i = (cfg.INTERCEPTOR["fuel"] / cfg.INTERCEPTOR["burn_rate"]) if cfg.INTERCEPTOR["burn_rate"] > 0 else 0.0
    t_end = burn_time_i + 10
    t_points = np.linspace(t_start, t_end, 200)

    sim = sim_mod.Simulation(interceptor_initial, target_initial)
    solution = sim.run(t_span=(t_start, t_end), t_eval=t_points)

    # --- metrics (same calculations as main.py) ---
    times_full = solution.t
    vel_i_full = solution.y[3:6, :]
    fuel_full = solution.y[6, :]

    fuel_initial = float(fuel_full[0])
    fuel_remaining = float(fuel_full[-1])
    fuel_used = max(0.0, fuel_initial - max(0.0, fuel_remaining))

    if times_full.size >= 2:
        a_components = np.gradient(vel_i_full, times_full, axis=1)
        accel_mag = np.linalg.norm(a_components, axis=0)
        max_accel = float(np.max(accel_mag))
        duration = float(times_full[-1] - times_full[0])
        # trapezoid was renamed from trapz in newer numpy
        _trapz = getattr(np, 'trapezoid', np.trapz)
        avg_accel = float((_trapz(accel_mag, times_full) / duration) if duration > 0 else 0.0)
    else:
        max_accel = 0.0
        avg_accel = 0.0

    # compute separations
    diffs = solution.y[7:10, :] - solution.y[0:3, :]
    dists = np.linalg.norm(diffs, axis=0)
    min_idx = int(np.argmin(dists))
    min_sep = float(dists[min_idx])

    # failure conditions
    away_failed, away_time = detect_away_timeout(solution.t, dists, cfg.AWAY_TIMEOUT_SEC)
    floor_failed = bool(getattr(solution, 'floor_miss', False))
    floor_time = float(getattr(solution, 'floor_miss_time', np.nan)) if floor_failed else None

    collided = False
    collision_time = None
    if hasattr(solution, 't_events') and solution.t_events and len(solution.t_events) > 0 and len(solution.t_events[0]) > 0:
        collided = True
        collision_time = float(solution.t_events[0][0])

    return {
        "solution": solution,
        "interceptor_initial": interceptor_initial,
        "target_initial": target_initial,
        "collided": collided, "collision_time": collision_time,
        "min_sep": min_sep,
        "away_failed": away_failed, "away_time": away_time,
        "floor_failed": floor_failed, "floor_time": floor_time,
        "fuel_used": fuel_used,
        "max_accel": max_accel, "avg_accel": avg_accel,
    }


def truncate_trajectories(res):
    # same truncation priority as main.py: floor > away > collision
    solution = res["solution"]
    i_traj = solution.y[0:3, :].copy()
    t_traj = solution.y[7:10, :].copy()
    times = solution.t.copy()

    collided = res["collided"]
    collision_time = res["collision_time"]
    floor_failed = res["floor_failed"]
    floor_time = res["floor_time"]
    away_failed = res["away_failed"]
    away_time = res["away_time"]

    if floor_failed and (collision_time is None or floor_time < collision_time) and (not away_failed or floor_time < away_time):
        idx_cut = int((np.abs(times - floor_time)).argmin())
        i_traj = i_traj[:, :idx_cut + 1]
        t_traj = t_traj[:, :idx_cut + 1]
        times = times[:idx_cut + 1]
        collided = False
        away_failed = False
    elif away_failed and (collision_time is None or away_time < collision_time):
        idx_cut = int((np.abs(times - away_time)).argmin())
        i_traj = i_traj[:, :idx_cut + 1]
        t_traj = t_traj[:, :idx_cut + 1]
        times = times[:idx_cut + 1]
        collided = False
    elif collided and collision_time is not None:
        idx_col = int((np.abs(times - collision_time)).argmin())
        i_traj = i_traj[:, :idx_col + 1]
        t_traj = t_traj[:, :idx_col + 1]
        times = times[:idx_col + 1]

    return i_traj, t_traj, times, collided, floor_failed, away_failed


def build_plotly_figure(res, guidance_law, target_name):
    # interactive 3d plot using plotly instead of matplotlib
    # same data and axis mapping as main.py (y/z swapped for display)
    solution = res["solution"]
    i_traj, t_traj, times, collided, floor_failed, away_failed = truncate_trajectories(res)
    collision_time = res["collision_time"]
    min_sep = res["min_sep"]

    fig = go.Figure()

    hover_times = [f"t={t:.1f}s" for t in times]

    # plot trajectories (swap y and z same as main.py)
    fig.add_trace(go.Scatter3d(
        x=i_traj[0], y=i_traj[2], z=i_traj[1],
        mode='lines', name='Interceptor Trajectory',
        line=dict(color='#2563eb', width=4),
        text=hover_times, hoverinfo='text+name',
    ))
    fig.add_trace(go.Scatter3d(
        x=t_traj[0], y=t_traj[2], z=t_traj[1],
        mode='lines', name='Target Trajectory',
        line=dict(color='#dc2626', width=4),
        text=hover_times, hoverinfo='text+name',
    ))

    # start and end points
    ii = res["interceptor_initial"]
    ti = res["target_initial"]
    fig.add_trace(go.Scatter3d(
        x=[ii['position'][0]], y=[ii['position'][2]], z=[ii['position'][1]],
        mode='markers', name='Interceptor Start',
        marker=dict(color='#2563eb', size=6, symbol='circle'),
    ))
    fig.add_trace(go.Scatter3d(
        x=[ti['position'][0]], y=[ti['position'][2]], z=[ti['position'][1]],
        mode='markers', name='Target Start',
        marker=dict(color='#dc2626', size=6, symbol='circle'),
    ))
    fig.add_trace(go.Scatter3d(
        x=[i_traj[0, -1]], y=[i_traj[2, -1]], z=[i_traj[1, -1]],
        mode='markers', name='Interceptor End',
        marker=dict(color='#2563eb', size=5, symbol='x'),
    ))
    fig.add_trace(go.Scatter3d(
        x=[t_traj[0, -1]], y=[t_traj[2, -1]], z=[t_traj[1, -1]],
        mode='markers', name='Target End',
        marker=dict(color='#dc2626', size=5, symbol='x'),
    ))

    # collision/failure markers
    if collided and collision_time is not None:
        idx_col = int((np.abs(solution.t - collision_time)).argmin())
        r_i_col = solution.y[0:3, idx_col]
        r_t_col = solution.y[7:10, idx_col]
        cp = (r_i_col + r_t_col) / 2.0
        fig.add_trace(go.Scatter3d(
            x=[cp[0]], y=[cp[2]], z=[cp[1]],
            mode='markers+text', name='Collision',
            marker=dict(color='#16a34a', size=8, symbol='diamond'),
            text=[f"Hit @ {collision_time:.2f}s<br>sep~{min_sep:.1f}m"],
            textposition='top center', textfont=dict(color='#16a34a', size=12),
        ))
    elif floor_failed and len(times) > 0:
        fig.add_trace(go.Scatter3d(
            x=[i_traj[0, -1]], y=[i_traj[2, -1]], z=[i_traj[1, -1]],
            mode='markers', name='Failed (floor contact)',
            marker=dict(color='black', size=7, symbol='x'),
        ))
    elif away_failed and len(times) > 0:
        fig.add_trace(go.Scatter3d(
            x=[i_traj[0, -1]], y=[i_traj[2, -1]], z=[i_traj[1, -1]],
            mode='markers', name='Failed (moving away)',
            marker=dict(color='black', size=7, symbol='x'),
        ))

    # time labels every 10 seconds
    last_time = solution.t[-1]
    label_times = np.arange(10, last_time, 10)
    for lt in label_times:
        idx = int((np.abs(solution.t - lt)).argmin())
        if idx < i_traj.shape[1] and idx < t_traj.shape[1]:
            fig.add_trace(go.Scatter3d(
                x=[i_traj[0, idx]], y=[i_traj[2, idx]], z=[i_traj[1, idx]],
                mode='text', text=[f"{int(round(solution.t[idx]))}s"],
                textfont=dict(color='#2563eb', size=10), showlegend=False,
            ))
            fig.add_trace(go.Scatter3d(
                x=[t_traj[0, idx]], y=[t_traj[2, idx]], z=[t_traj[1, idx]],
                mode='text', text=[f"{int(round(solution.t[idx]))}s"],
                textfont=dict(color='#dc2626', size=10), showlegend=False,
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X position (Crossrange, m)',
            yaxis_title='Z position (Downrange, m)',
            zaxis_title='Y position (Altitude, m)',
            aspectmode='data',
        ),
        title=dict(text=f"Engagement Simulation ({guidance_law.upper()})",
                   x=0.5, y=0.98, yanchor='top'),
        height=700,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(orientation='h', yanchor='top', y=-0.02, xanchor='center', x=0.5),
    )
    return fig


def build_matplotlib_figure(res, guidance_law):
    # renders the same plot as main.py using matplotlib, returns png bytes
    solution = res["solution"]
    i_traj, t_traj, times, collided, floor_failed, away_failed = truncate_trajectories(res)
    collision_time = res["collision_time"]
    min_sep = res["min_sep"]
    ii = res["interceptor_initial"]
    ti = res["target_initial"]

    figure = mpl.figure(figsize=(16, 12))
    ax = figure.add_subplot(111, projection='3d')

    # plot trajectories (swap y and z for correct axis mapping)
    ax.plot(i_traj[0], i_traj[2], i_traj[1], label='Interceptor Trajectory', color='b')
    ax.plot(t_traj[0], t_traj[2], t_traj[1], label='Target Trajectory', color='r')

    # time labels every 10s
    last_time = solution.t[-1]
    label_times = np.arange(10, last_time, 10)
    for label_time in label_times:
        idx = int((np.abs(solution.t - label_time)).argmin())
        if idx < i_traj.shape[1] and idx < t_traj.shape[1]:
            ax.text(i_traj[0, idx], i_traj[2, idx], i_traj[1, idx],
                    f"{int(round(solution.t[idx]))}s", color='blue', fontsize=10, ha='center', va='center')
            ax.text(t_traj[0, idx], t_traj[2, idx], t_traj[1, idx],
                    f"{int(round(solution.t[idx]))}s", color='red', fontsize=10, ha='center', va='center')

    # mark start and end points (swap y and z)
    ax.scatter(ii['position'][0], ii['position'][2], ii['position'][1],
               color='blue', marker='o', s=100, label='Interceptor Start')
    ax.scatter(ti['position'][0], ti['position'][2], ti['position'][1],
               color='red', marker='o', s=100, label='Target Start')
    ax.scatter(i_traj[0, -1], i_traj[2, -1], i_traj[1, -1],
               color='blue', marker='x', s=100, label='Interceptor End')
    ax.scatter(t_traj[0, -1], t_traj[2, -1], t_traj[1, -1],
               color='red', marker='x', s=100, label='Target End')

    ax.set_xlabel('X position (Crossrange, m)')
    ax.set_ylabel('Z position (Downrange, m)')
    ax.set_zlabel('Y position (Altitude, m)')
    ax.set_title(f'Engagement Simulation ({guidance_law.upper()})')

    # collision/away markers and annotation
    if collided and collision_time is not None:
        idx_col = int((np.abs(solution.t - collision_time)).argmin())
        r_i_col = solution.y[0:3, idx_col]
        r_t_col = solution.y[7:10, idx_col]
        collision_point = (r_i_col + r_t_col) / 2.0
        ax.scatter(collision_point[0], collision_point[2], collision_point[1],
                   color='green', marker='*', s=200, label='Collision')
        ax.text(collision_point[0], collision_point[2], collision_point[1],
                f"Hit @ {collision_time:.2f}s\nsep~{min_sep:.1f}m",
                color='green', fontsize=10, ha='left', va='bottom')
    elif floor_failed and len(times) > 0:
        ax.scatter(i_traj[0, -1], i_traj[2, -1], i_traj[1, -1],
                   color='black', marker='x', s=140, label='Failed (floor contact)')
    elif away_failed and len(times) > 0:
        ax.scatter(i_traj[0, -1], i_traj[2, -1], i_traj[1, -1],
                   color='black', marker='x', s=120, label='Failed (moving away)')

    ax.legend()
    ax.grid(True)

    # status text at bottom
    status_text = (
        f"Collision: {collided} | Min separation: {min_sep:.2f} m | Threshold: {cfg.INTERCEPTION_RADIUS:.2f} m"
        + (f" | Failed: floor contact after 3.0s" if floor_failed and not collided else "")
        + (f" | Failed: moving away > {cfg.AWAY_TIMEOUT_SEC:.1f}s since CPA" if away_failed and not collided and not floor_failed else "")
    )
    figure.text(0.02, 0.02, status_text, fontsize=12, color=('green' if collided else 'red'))

    # set axis limits based on trajectory extents
    x_min = min(i_traj[0].min(), t_traj[0].min())
    x_max = max(i_traj[0].max(), t_traj[0].max())
    y_min = min(i_traj[2].min(), t_traj[2].min())
    y_max = max(i_traj[2].max(), t_traj[2].max())
    z_min = min(i_traj[1].min(), t_traj[1].min())
    z_max = max(i_traj[1].max(), t_traj[1].max())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    buf = io.BytesIO()
    figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    mpl.close(figure)
    buf.seek(0)
    return buf.getvalue()


# *** run and display results ***

if run_clicked:
    with st.spinner("Running simulation..."):
        res = run_simulation(guidance_law, target_name, seed)
    st.session_state["last_result"] = res
    st.session_state["last_guidance"] = guidance_law
    st.session_state["last_target"] = target_name

if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    gl = st.session_state["last_guidance"]
    tn = st.session_state["last_target"]

    st.markdown("---")

    # outcome banner
    if res["collided"]:
        st.success(f"**INTERCEPTION** — Hit at **{res['collision_time']:.2f}s** | Min separation: **{res['min_sep']:.1f} m**")
    elif res["floor_failed"]:
        st.error(f"**MISS (floor contact)** — Min separation: **{res['min_sep']:.1f} m**")
    elif res["away_failed"]:
        st.error(f"**MISS (moving away)** — Min separation: **{res['min_sep']:.1f} m**")
    else:
        st.warning(f"**NO HIT** — Min separation: **{res['min_sep']:.1f} m**")

    # metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Min Separation", f"{res['min_sep']:.1f} m")
    m2.metric("Fuel Used", f"{res['fuel_used']:.1f} kg")
    m3.metric("Max Acceleration", f"{res['max_accel']:.1f} m/s\u00b2")
    m4.metric("Avg Acceleration", f"{res['avg_accel']:.1f} m/s\u00b2")

    # 3d plot
    fig = build_plotly_figure(res, gl, tn)
    st.plotly_chart(fig, use_container_width=True, key="plotly_3d")

    # separation distance over time
    solution = res["solution"]
    diffs = solution.y[7:10, :] - solution.y[0:3, :]
    dists = np.linalg.norm(diffs, axis=0)

    sep_fig = go.Figure()
    sep_fig.add_trace(go.Scatter(
        x=solution.t, y=dists, mode='lines', name='Separation',
        line=dict(color='#7c3aed', width=2),
    ))
    sep_fig.add_hline(y=cfg.INTERCEPTION_RADIUS, line_dash="dash", line_color="#16a34a",
                      annotation_text=f"Interception radius ({cfg.INTERCEPTION_RADIUS} m)")
    sep_fig.update_layout(
        title="Separation Distance Over Time",
        xaxis_title="Time (s)", yaxis_title="Distance (m)",
        height=350, margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(sep_fig, use_container_width=True, key="plotly_sep")

    # optional matplotlib version
    st.markdown("---")
    show_mpl = st.checkbox("Show original Matplotlib plot")
    if show_mpl:
        with st.spinner("Rendering Matplotlib plot..."):
            png_bytes = build_matplotlib_figure(res, gl)
        st.image(png_bytes, caption=f"Matplotlib — Engagement Simulation ({gl.upper()})")

else:
    st.markdown(
        "<div style='text-align:center; padding:60px 20px; color:grey;'>"
        "<h3>Select a guidance law and target scenario, then press <b>Run Simulation</b></h3>"
        "<p>The simulator models a ground-launched interceptor engaging an airborne drone target in 3D space.<br>"
        "It compares three guidance strategies: Pure Pursuit, Proportional Navigation, and Augmented PN.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
