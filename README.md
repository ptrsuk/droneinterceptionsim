# Drone Interception Simulator

3D physics-based simulator developed in Python for comparative performance analysis of three guidance laws for drone interception: **Pure Pursuit (PP)**, **Proportional Navigation (PN)**, and **Augmented Proportional Navigation (APN)**.

## Setup

Designed for **Python 3.10**. Requires NumPy, Matplotlib, PyTest, Streamlit, and Plotly.

```
pip install -r requirements.txt
```

## Running the Simulation

### Option 1 — Web UI (recommended)

Run from the project root:
```
streamlit run src/app.py
```
Open the URL shown in your browser (usually `http://localhost:8501`).
Select a guidance law, target scenario, and seed from the sidebar, then click **Run Simulation**.
An interactive 3D plot and metrics will be displayed.

### Option 2 — Single Engagement (CLI)

Configure the scenario in `src/config.py` — set the `GUIDANCE_LAW` and `TARGET_TEMPLATE_NAME` variables.
Target template list can be found in `src/target_templates.py`.
Run `src/main.py` from the terminal or IDE. An interactive 3D plot will be displayed.

### Option 3 — Batch Run

Configure `src/batch_run_main.py`:
- Set `TARGET_ORDER` with the targets and order you want to run them in
- Set `BATCH_PLAN` with the target templates and number of seeds for each

Run `src/batch_run_main.py` from the terminal or IDE.
The batch runner will create a `batch_out/` folder with PNGs and a compiled CSV file.

## Unit Tests

```
pytest unit_tests/unit_tests.py -v
```

---

Peter Simpson, September 2025
