"""How much avoidance does the collision optimiser apply, before and after the fix?

THE QUESTION
------------
Correcting the coordinate truncation improves open-loop L2. Li et al. (2024) showed that
open-loop L2 rewards trajectories that stay close to the recorded human path, so any such gain
invites the reply: "you did not plan better, you merely avoided less." Every answer we have so
far is indirect -- the mirror control shows the displacement the fix removes is DIRECTIONAL,
and a full-split measurement shows forward progress is unchanged while lateral swing falls.
Neither measures the avoidance itself.

This script measures it directly. With the collision optimiser switched off (R-0084) the
trajectory is the network's raw output, so

    ||arm - raw||

is precisely the displacement the optimiser applies on that frame. Comparing that magnitude
between the published arm and the corrected one turns the objection into a number.

READING IT
----------
If the corrected arm's avoidance is SMALLER, that is the honest finding and we report it: the
planner avoids less because it is no longer avoiding obstacles reported half a metre from
where they are. The claim then rests on the mirror control, which says the displacement
removed was aimed, plus the collision counts, which did not worsen.
If the corrected arm's avoidance is the SAME SIZE but differently directed, the objection
dissolves outright.
Either way the number is more use than the argument. It is reported per horizon, paired, with
bootstrap intervals, on the frames where the two arms differ.
"""
import argparse
import glob
import json

import numpy as np

from analyse_quantfix import HOR, boot, load  # noqa: F401  (HOR/boot reused)
import mirror_control as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="work_dirs/R-0084_*_nocoloptim")
    ap.add_argument("--base", default="work_dirs/R-0072_2026-08-17_baseline_full")
    ap.add_argument("--arm", default="work_dirs/R-0073_2026-08-17_quantfix_full")
    ap.add_argument("--out", default="results/avoidance.json")
    a = ap.parse_args()

    raw_dir = sorted(glob.glob(a.raw))[0]
    R = M.load_traj(raw_dir + "/results.pkl")
    B = M.load_traj(a.base + "/results.pkl")
    F = M.load_traj(a.arm + "/results.pkl")
    assert R.shape == B.shape == F.shape, "arms differ in length"
    print(f"raw (no optimiser) : {raw_dir.split('/')[-1]}")
    print(f"published          : {a.base.split('/')[-1]}")
    print(f"corrected          : {a.arm.split('/')[-1]}")

    moved = np.abs(F - B).max(axis=(1, 2)) > 1e-9
    print(f"\nframes where the correction changes the trajectory: {moved.sum()} of {len(R)}\n")

    dB = np.linalg.norm(B - R, axis=2)      # (N, 6) avoidance applied, published
    dF = np.linalg.norm(F - R, axis=2)      # (N, 6) avoidance applied, corrected

    res = {"raw": raw_dir.split("/")[-1], "n_moved": int(moved.sum())}
    print("magnitude of the displacement the collision optimiser applies (m)")
    print(f"{'horizon':>9}{'published':>12}{'corrected':>12}{'delta':>11}   95% CI")
    for h, t in enumerate(HOR):
        x, y = dB[moved, h], dF[moved, h]
        d = y - x
        lo, hi = boot(d)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"{t:8.1f}s{x.mean():12.4f}{y.mean():12.4f}{d.mean():+11.4f}   "
              f"[{lo:+.4f},{hi:+.4f}]{sig}")
        res[f"avoid@{t}"] = {"published": float(x.mean()), "corrected": float(y.mean()),
                             "delta": float(d.mean()), "ci": [float(lo), float(hi)],
                             "significant": bool(lo > 0 or hi < 0)}

    # totals over the horizon, and how often the correction increases avoidance
    tb, tf = dB[moved].mean(1), dF[moved].mean(1)
    lo, hi = boot(tf - tb)
    frac_up = float((tf > tb).mean())
    print(f"\n{'mean over horizons':>22}{tb.mean():12.4f}{tf.mean():12.4f}"
          f"{(tf-tb).mean():+11.4f}   [{lo:+.4f},{hi:+.4f}]")
    print(f"frames where the correction AVOIDS MORE than the published arm: {100*frac_up:.1f}%")
    print("  (a pure 'avoids less' story would put this near zero; a directional change"
          " leaves it substantial)")
    res["mean_over_horizons"] = {"published": float(tb.mean()), "corrected": float(tf.mean()),
                                 "delta": float((tf - tb).mean()), "ci": [float(lo), float(hi)],
                                 "frac_frames_avoiding_more": frac_up}

    # how much of the change is a change of size versus a change of direction
    dsize = np.abs(tf - tb)                                   # change in magnitude
    dvec = np.linalg.norm(F[moved] - B[moved], axis=2).mean(1)  # total change
    print(f"\ntotal change in the applied displacement : {dvec.mean():.4f} m")
    print(f"  of which a change in its magnitude      : {dsize.mean():.4f} m "
          f"({100*dsize.mean()/max(dvec.mean(),1e-9):.0f}%)")
    print("  the remainder is a change of direction at unchanged size")
    res["change_decomposition"] = {"total": float(dvec.mean()), "magnitude_part": float(dsize.mean())}

    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
