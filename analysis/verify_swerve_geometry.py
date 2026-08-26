"""Does the corrected planner travel less, or swerve less?

Section 7 answers the "it just deviates less" objection partly by pointing out that the
correction does not shorten the path: forward progress is essentially unchanged while
peak lateral excursion drops. That distinction matters -- travelling less would be a
conservatism effect, swerving less is what a corrected obstacle position should produce
-- so the two numbers are re-derived here rather than left in console output.

Measured over the frames the correction actually moves, since on the rest the two arms
are identical and including them only dilutes the effect toward zero:

    forward progress        mean forward coordinate of the final (3 s) waypoint
    path length             mean cumulative length from the ego origin
    peak lateral excursion  mean over frames of the largest |lateral| over waypoints

Component 0 of `planning_traj` is lateral and component 1 is forward, matching the axes
of the qualitative figure.

Usage (from the root of the UniAD working tree):

    python3 analysis/verify_swerve_geometry.py --out results/swerve_geometry.json
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


def path_length(t):
    withorigin = np.concatenate([np.zeros((len(t), 1, 2)), t], axis=1)
    return np.linalg.norm(np.diff(withorigin, axis=1), axis=2).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="work_dirs/R-0072_2026-08-17_baseline_full")
    ap.add_argument("--arm", default="work_dirs/R-0073_2026-08-17_quantfix_full")
    ap.add_argument("--out", default="results/swerve_geometry.json")
    a = ap.parse_args()

    B = load_traj(a.base + "/results.pkl")
    F = load_traj(a.arm + "/results.pkl")
    moved = np.abs(F - B).max(axis=(1, 2)) > 1e-9

    out = {"n_frames_total": int(len(B)), "n_frames_moved": int(moved.sum())}
    for tag, sel in (("moved_frames", moved), ("all_frames", slice(None))):
        b, f = B[sel], F[sel]
        fwd_b, fwd_f = float(b[:, -1, 1].mean()), float(f[:, -1, 1].mean())
        len_b, len_f = float(path_length(b).mean()), float(path_length(f).mean())
        lat_b = float(np.abs(b[:, :, 0]).max(1).mean())
        lat_f = float(np.abs(f[:, :, 0]).max(1).mean())
        out[tag] = {
            "forward_progress_baseline_m": fwd_b,
            "forward_progress_corrected_m": fwd_f,
            "forward_progress_change_m": fwd_f - fwd_b,
            "path_length_baseline_m": len_b,
            "path_length_corrected_m": len_f,
            "path_length_change_m": len_f - len_b,
            "peak_lateral_baseline_m": lat_b,
            "peak_lateral_corrected_m": lat_f,
            "peak_lateral_change_pct": 100 * (lat_f - lat_b) / lat_b,
        }
        print("%-12s forward %+.4f m   path %+.4f m   peak lateral %+.2f%%"
              % (tag, fwd_f - fwd_b, len_f - len_b,
                 out[tag]["peak_lateral_change_pct"]))

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
