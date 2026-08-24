"""Extract S_global/S_trig/S_risk PlanningMetric results from a results.pkl
produced by custom_multi_gpu_test (see uniad/apis/test.py) into metrics.json,
matching the manual convention used for prior runs (e.g. R-0002_2026-07-29).

Usage: python3 build_metrics_json.py <run_dir> <run_id> <config_path>
"""
import sys
import json
import mmcv

HORIZONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def to_list(v):
    if v is None:
        return None
    if hasattr(v, "cpu"):
        return v.cpu().tolist()
    return list(v)


def tier_dict(tier, n_frames):
    if tier is None:
        return None
    return {
        "n_frames": n_frames,
        "obj_col": to_list(tier["obj_col"]),
        "obj_box_col": to_list(tier["obj_box_col"]),
        "L2": to_list(tier["L2"]),
    }


def main():
    run_dir, run_id, config_path = sys.argv[1], sys.argv[2], sys.argv[3]
    results = mmcv.load(f"{run_dir}/results.pkl")

    n_trig = results.get("n_trig", 0)
    n_risk = results.get("n_risk", 0)
    n_total = results.get("n_total")

    out = {
        "run_id": run_id,
        "config": config_path,
        "metrics_horizons_s": HORIZONS,
        "n_total": n_total,
        "n_trig": n_trig,
        "n_risk": n_risk,
        "S_global": tier_dict(results.get("planning_results_computed"), n_total),
        "S_trig": tier_dict(results.get("planning_results_computed_trig"), n_trig),
        "S_risk": tier_dict(results.get("planning_results_computed_risk"), n_risk),
    }

    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    per_frame = results.get("planning_per_frame")
    if per_frame:
        with open(f"{run_dir}/per_frame.json", "w") as f:
            json.dump(per_frame, f)
        print(f"wrote {run_dir}/per_frame.json  n_frames={len(per_frame)} (for bootstrap CI)")

    latency = results.get("ood_latency_profile")
    if latency:
        with open(f"{run_dir}/latency_profile.json", "w") as f:
            json.dump(latency, f)
        print(f"wrote {run_dir}/latency_profile.json  n_frames={len(latency)} "
              f"(safety-layer inference overhead; all-zero unless PROFILE_OOD_LATENCY=1 was set)")

    print(f"wrote {run_dir}/metrics.json  n_total={n_total} n_trig={n_trig} n_risk={n_risk}")


if __name__ == "__main__":
    main()
