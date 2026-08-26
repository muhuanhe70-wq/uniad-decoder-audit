"""How far apart are the exact-cast and rounding arms, frame by frame?

Section 5.2 argues that the rounding ablation is informative precisely because it is
not the same run under another name: it reproduces the aggregate gain to within 0.8 mm
while moving individual trajectories substantially. That contrast needs a number for
"substantially", and the number depends on how it is measured, so the definition is
fixed here rather than left implicit.

Reported, over the frames on which the two arms differ at all:

    per-waypoint Euclidean distance        mean and max over all (frame, waypoint)
    per-frame maximum waypoint distance    mean and max over frames   <- quoted
    per-axis absolute difference           mean and max, for comparison

The per-frame maximum is what the paper quotes: on a typical differing frame, the
waypoint that moves most moves by that much. Euclidean is used because every other
displacement in the paper is Euclidean; the per-axis figures are printed alongside so
the choice is visible.

Usage (from the root of the UniAD working tree):

    python3 analysis/verify_rounding_divergence.py --out results/rounding_divergence.json
"""
import argparse
import gc
import io
import json

import mmcv
import numpy as np
import torch

torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")


def load_traj(path):
    b = mmcv.load(path)["bbox_results"]
    tr = np.stack([f["planning_traj"].detach().cpu().numpy()[0]
                   for f in b]).astype(np.float64)
    del b
    gc.collect()
    return tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", default="work_dirs/R-0073_2026-08-17_quantfix_full")
    ap.add_argument("--round", dest="rnd",
                    default="work_dirs/R-0083_2026-08-22_quantround")
    ap.add_argument("--out", default="results/rounding_divergence.json")
    a = ap.parse_args()

    F = load_traj(a.exact + "/results.pkl")
    R = load_traj(a.rnd + "/results.pkl")
    assert F.shape == R.shape

    differ = np.abs(F - R).max(axis=(1, 2)) > 1e-9
    d = np.linalg.norm(F - R, axis=2)[differ]      # (n, 6) Euclidean per waypoint
    comp = np.abs(F - R)[differ]                   # (n, 6, 2) per axis

    out = {
        "n_frames_total": int(len(F)),
        "n_frames_differing": int(differ.sum()),
        "euclidean_per_waypoint_mean_m": float(d.mean()),
        "euclidean_per_waypoint_max_m": float(d.max()),
        "euclidean_per_frame_max_mean_m": float(d.max(1).mean()),
        "euclidean_per_frame_max_max_m": float(d.max(1).max()),
        "per_axis_per_frame_max_mean_m": float(comp.max((1, 2)).mean()),
        "per_axis_per_frame_max_max_m": float(comp.max((1, 2)).max()),
        "quoted_in_paper": "euclidean_per_frame_max_mean_m and euclidean_per_frame_max_max_m",
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("frames differing              %d of %d"
          % (out["n_frames_differing"], out["n_frames_total"]))
    print("per-frame max waypoint move   mean %.4f m, worst %.4f m  <- quoted"
          % (out["euclidean_per_frame_max_mean_m"],
             out["euclidean_per_frame_max_max_m"]))
    print("per-waypoint Euclidean        mean %.4f m, worst %.4f m"
          % (out["euclidean_per_waypoint_mean_m"], out["euclidean_per_waypoint_max_m"]))
    print("per-axis (older convention)   mean %.4f m, worst %.4f m"
          % (out["per_axis_per_frame_max_mean_m"], out["per_axis_per_frame_max_max_m"]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
