

# src/batch_run_main.py
# uses main.py to create PNGs and fill runs.csv in predefined loop
import os

import time
import csv
import copy
import importlib
from pathlib import Path
import numpy as np

import config as cfg
import target_templates as tt



GUIDANCE_DIGIT = {"pp": 1, "pn": 2, "apn": 3}

# target order - determines run_id
TARGET_ORDER = [
    "static_zero",
    "drone_hover",
    "drone_low_level_snake",
    "drone_zigzag_climb",
    "quadcopter_aggressive_rise",
    "prop_high_speed_cruise",
]

TARGET_DIGIT = {name: i+1 for i, name in enumerate(TARGET_ORDER)}
GUIDANCES = ["pp", "pn", "apn"]

# batch plan - target name, number of seeds
BATCH_PLAN = [
    ("static_zero", 1),
    ("drone_hover", 8),
    ("drone_low_level_snake", 8),
    ("drone_zigzag_climb", 8),
    ("quadcopter_aggressive_rise", 8),
    ("prop_high_speed_cruise", 8),
]

OUT_DIR = Path("batch_out")
PNG_DIR = OUT_DIR / "png"
CSV_PATH = OUT_DIR / "runs.csv"

def run_id(guidance, target, seed_idx):
    return 100*GUIDANCE_DIGIT[guidance] + 10*TARGET_DIGIT[target] + int(seed_idx)

def get_tuning_row(guidance: str):
    PP = getattr(cfg, "PP", {})
    PN = getattr(cfg, "PN", {})
    APN = getattr(cfg, "APN", {})
    if guidance == "pp":
        return {
            "N": None, "aug_gain": None, "acc_tau": None, "blend_pp": None,
            "damping": PP.get("damping"), "forward_bias": None, "lat_gain": None
        }
    elif guidance == "pn":
        return {
            "N": PN.get("N"), "aug_gain": None, "acc_tau": None, "blend_pp": PN.get("blend_pp"),
            "damping": PN.get("damping"), "forward_bias": PN.get("forward_bias"), "lat_gain": PN.get("lat_gain")
        }
    else:
        return {
            "N": APN.get("N"), "aug_gain": APN.get("aug_gain"), "acc_tau": APN.get("acc_tau"), "blend_pp": APN.get("blend_pp"),
            "damping": APN.get("damping"), "forward_bias": APN.get("forward_bias"), "lat_gain": APN.get("lat_gain")
        }

def main():
    start_time = time.time()

    os.environ["SIM_SAVE_PNG"] = "1"
    os.environ["BATCH_MODE"] = "1"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "target_name", "guidance_law", "run_id",
        "seed", "success", "time_to_hit", "min_separation", "fail_type",
        "fuel_used_kg", "max_accel_mps2", "avg_accel_mps2",
        "N", "aug_gain", "acc_tau", "blend_pp", "damping", "forward_bias", "lat_gain",
        "png_path"
    ]

    rows = []
    for target_name, nseeds in BATCH_PLAN:
        for guidance in GUIDANCES:
            # determine base seed
            base_seed = int(tt.TARGET_TEMPLATES[target_name].get("maneuver", {}).get("seed", 1000))
            for idx in range(1, nseeds+1):
                seed_val = base_seed + (idx-1)
                rid = run_id(guidance, target_name, idx)
                png_name = f"{target_name}_{guidance}_seed{idx}_ID{rid:03d}.png"
                png_path = (PNG_DIR / png_name).as_posix()

                cfg.GUIDANCE_LAW = guidance
                cfg.TARGET_TEMPLATE_NAME = target_name
                template = tt.TARGET_TEMPLATES[target_name]
                target_cfg = copy.deepcopy(template)
                if isinstance(target_cfg.get("maneuver"), dict) and target_cfg["maneuver"].get("type") == "noisy":
                    target_cfg["maneuver"]["seed"] = int(seed_val)
                cfg.TARGET = target_cfg
                cfg.TARGET_INITIAL = {
                    "position": target_cfg["position"],
                    "velocity": target_cfg["velocity"],
                }

                # reload modules with constants - only way to make sure the constants are loaded the same every
                import kinematics as kine
                import simulation as sim
                import main as main_mod
                importlib.reload(kine)
                importlib.reload(sim)
                importlib.reload(main_mod)

                title_suffix = f"| ID {rid:03d}"

                result = main_mod.main(save_path=png_path, title_suffix=title_suffix)

                tuning = get_tuning_row(guidance)
                rows.append({
                    # basic info
                    "target_name": target_name,
                    "guidance_law": guidance,
                    "run_id": f"{rid:03d}",
                    "seed": seed_val,

                    # results
                    "success": result.get("success", False),
                    "time_to_hit": result.get("time_to_hit"),
                    "min_separation": result.get("min_separation"),
                    "fail_type": result.get("fail_type", "none"),
                    "fuel_used_kg": result.get("fuel_used_kg"),
                    "max_accel_mps2": result.get("max_accel_mps2"),
                    "avg_accel_mps2": result.get("avg_accel_mps2"),

                    # tuning parameters
                    "N": tuning["N"],
                    "aug_gain": tuning["aug_gain"],
                    "acc_tau": tuning["acc_tau"],
                    "blend_pp": tuning["blend_pp"],
                    "damping": tuning["damping"],
                    "forward_bias": tuning["forward_bias"],
                    "lat_gain": tuning["lat_gain"],

                    "png_path": png_path,
                })

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Wrote {len(rows)} rows to {CSV_PATH}")
    print(f"PNGs saved to {PNG_DIR}")
    print(f"Batch run completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
