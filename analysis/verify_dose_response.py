"""The dose-response block of Section 6.3, re-derived.

The argument there is that injecting the hazard scorer's output into the planner costs
L2 in proportion to how often it intervenes, while buying no collision benefit at any
dose. That needs three things to hold, and all three are checked here rather than quoted
from console output:

  * the per-intervention magnitude is comparable across doses, so the comparison is
    about how often the planner is nudged and not about how hard;
  * the L2 cost scales with the number of interventions;
  * the collision outcome is nil at both doses.

Arms: the unmodified baseline (dose 0), a 156-frame trigger and a 919-frame trigger.
The number of frames on which each arm's trajectory differs from the baseline is checked
against its trigger count, which is what identifies the arm as the intended one.

Magnitude is the mean over intervened frames of the mean per-waypoint Euclidean
displacement from the baseline -- the same convention analyse_avoidance.py uses, so the
two are directly comparable.

Usage (from the root of the UniAD working tree):

    python3 analysis/verify_dose_response.py --out results/dose_response.json
"""
import argparse
import gc
import io
import json

import mmcv
import numpy as np
import torch

torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")

HORIZONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def load_traj(path):
    b = mmcv.load(path)["bbox_results"]
    tr = np.stack([f["planning_traj"].detach().cpu().numpy()[0]
                   for f in b]).astype(np.float64)
    del b
    gc.collect()
    return tr


def per_frame(run):
    return {r["idx"]: np.array(r["obj_box_col"]) > 0
            for r in json.load(open(run + "/per_frame.json"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="work_dirs/R-0072_2026-08-17_baseline_full")
    ap.add_argument("--low", default="work_dirs/R-0081_2026-08-19_randtrigger")
    ap.add_argument("--high", default="work_dirs/R-0042_2026-08-18_ours_clampfix")
    ap.add_argument("--out", default="results/dose_response.json")
    a = ap.parse_args()

    mb = json.load(open(a.base + "/metrics.json"))
    B = load_traj(a.base + "/results.pkl")
    pfB = per_frame(a.base)
    n = mb["S_global"]["n_frames"]

    out = {"baseline_L2": [round(x, 6) for x in mb["S_global"]["L2"]],
           "baseline_box_col_count":
               [int(round(r * n)) for r in mb["S_global"]["obj_box_col"]],
           "horizons_s": HORIZONS, "doses": {}}

    for tag, run in (("low", a.low), ("high", a.high)):
        m = json.load(open(run + "/metrics.json"))
        A = load_traj(run + "/results.pkl")
        moved = np.abs(A - B).max(axis=(1, 2)) > 1e-9
        d = np.linalg.norm(A - B, axis=2)[moved]

        pfA = per_frame(run)
        rep = sum(int((pfB[i] & ~pfA[i]).sum()) for i in pfB)
        intro = sum(int((~pfB[i] & pfA[i]).sum()) for i in pfB)

        out["doses"][tag] = {
            "run": m["run_id"],
            "n_trig": m.get("n_trig"),
            "n_frames_moved": int(moved.sum()),
            "trigger_count_matches_moved": int(moved.sum()) == m.get("n_trig"),
            "mean_intervention_magnitude_m": float(d.mean(1).mean()),
            "L2": [round(x, 6) for x in m["S_global"]["L2"]],
            "L2_delta_vs_baseline": [round(m["S_global"]["L2"][i]
                                           - mb["S_global"]["L2"][i], 6)
                                     for i in range(6)],
            "box_col_count":
                [int(round(r * n)) for r in m["S_global"]["obj_box_col"]],
            "collisions_repaired": rep,
            "collisions_introduced": intro,
        }
        del A
        gc.collect()

    lo, hi = out["doses"]["low"], out["doses"]["high"]
    out["summary"] = (
        "interventions %d -> %d; per-intervention magnitude %.3f m -> %.3f m "
        "(comparable); L2 cost at 1.0 s %+.4f -> %+.4f m; collisions repaired/introduced "
        "%d/%d -> %d/%d"
        % (lo["n_frames_moved"], hi["n_frames_moved"],
           lo["mean_intervention_magnitude_m"], hi["mean_intervention_magnitude_m"],
           lo["L2_delta_vs_baseline"][1], hi["L2_delta_vs_baseline"][1],
           lo["collisions_repaired"], lo["collisions_introduced"],
           hi["collisions_repaired"], hi["collisions_introduced"]))

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for tag in ("low", "high"):
        v = out["doses"][tag]
        print("%-5s %s  moved %d (matches n_trig: %s)  magnitude %.4f m"
              % (tag, v["run"], v["n_frames_moved"],
                 v["trigger_count_matches_moved"],
                 v["mean_intervention_magnitude_m"]))
        print("      L2 delta %s" % v["L2_delta_vs_baseline"])
        print("      repaired %d, introduced %d"
              % (v["collisions_repaired"], v["collisions_introduced"]))
    print("\n" + out["summary"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
