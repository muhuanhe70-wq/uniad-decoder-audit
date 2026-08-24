"""Does fixing UniAD's occupancy-quantisation defect change the planner's output?

THE DEFECT
----------
`planning_head.py::collision_optimization` writes the float result of
(index - bev/2) * 0.5 + 0.25 back into the int64 tensor `torch.nonzero` returns, so every
occupancy coordinate handed to the collision NLP is truncated towards zero. Verified verbatim
against the pristine OpenDriveLab source (a copy with zero OOD code, 248 lines against the
current 839), so it is UniAD's defect and not one this project introduced, and demonstrated
empirically rather than inferred: index (104, 103) should map to (2.25, 1.75) and the solver
receives (2, 1).

Measured over 200 indices: mean displacement 0.500 m, max 0.750 m, 50 % of cells off by at
least 0.5 m -- on a grid whose own resolution is 0.5 m. Every obstacle is pulled towards the
ego, so the repulsion field sits closer to the planned path than the obstacles actually are
and the planner should over-avoid. The prediction under test is therefore that correcting it
REDUCES L2 (less unnecessary deviation), with the collision rate as the quantity that must be
watched in case the accidental conservatism was buying something.

WHY BOTH ARMS ARE RERUN ON THE SAME FRAME SET
---------------------------------------------
Established 2026-08-17: running UniAD on the 847-frame risk subset does NOT reproduce the
full-split trajectories -- only 67.2 % of untriggered frames are bit-identical, mean
difference 0.033 m, max 6.68 m, because the subset breaks the temporal BEV queue. Comparing a
subset run against the full-split baseline is therefore confounded. Both arms here are run on
the identical ann_file so the only difference is the one-line cast.

Neither arm loads the OOD layer at all (enable_ood_injection=False), so this measures UniAD
against UniAD.
"""
import gc
import glob
import io
import json

import mmcv
import numpy as np
import torch
from scipy.stats import binomtest

torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")

HOR = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def load(pat):
    d = sorted(glob.glob(pat))[0]
    b = mmcv.load(d + "/results.pkl")["bbox_results"]
    tr = {f["token"]: f["planning_traj"].detach().cpu().numpy()[0].astype(np.float64) for f in b}
    order = [f["token"] for f in b]
    del b; gc.collect()
    pf = {r["idx"]: r for r in json.load(open(d + "/per_frame.json"))}
    return d, tr, order, pf


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    b = np.array([x[r.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return tuple(np.percentile(b, [2.5, 97.5]))


def main():
    import sys
    a = sys.argv[1:]
    pat0 = a[0] if len(a) > 0 else "work_dirs/R-0070_*_baseline_847"
    pat1 = a[1] if len(a) > 1 else "work_dirs/R-0071_*_quantfix_847"
    d0, T0, ord0, P0 = load(pat0)
    d1, T1, ord1, P1 = load(pat1)
    print(f"original UniAD : {d0.split('/')[-1]}")
    print(f"quantisation fixed: {d1.split('/')[-1]}")
    assert ord0 == ord1, "frame order differs; the runs are not comparable"
    n = len(ord0)
    print(f"paired on {n} identical frames, neither arm loads the OOD layer\n")

    diff = np.array([np.abs(T1[t] - T0[t]).max() for t in ord0])
    print(f"trajectories changed on {100*(diff > 1e-9).mean():.1f}% of frames "
          f"(mean max-abs change {diff.mean():.4f} m, p90 {np.percentile(diff,90):.4f} m)\n")

    print("L2 to the human trajectory  (negative = the fix is closer to the human)")
    print(f"{'horizon':>9}{'original':>11}{'fixed':>11}{'delta':>11}{'95% CI':>22}")
    res = {}
    for h, t in enumerate(HOR):
        a = np.array([P0[i]["L2"][h] for i in range(n)])
        b = np.array([P1[i]["L2"][h] for i in range(n)])
        dl = b - a
        lo, hi = boot(dl)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"{t:8.1f}s{a.mean():11.4f}{b.mean():11.4f}{dl.mean():+11.4f}"
              f"   [{lo:+.4f},{hi:+.4f}]{sig}")
        res[f"L2@{t}"] = {"orig": float(a.mean()), "fixed": float(b.mean()),
                          "delta": float(dl.mean()), "ci": [float(lo), float(hi)],
                          "significant": bool(lo > 0 or hi < 0)}

    print("\ncollision rate  (McNemar exact on discordant pairs)")
    for key, lab in (("obj_box_col", "box"), ("obj_col", "point")):
        print(f"  {lab}:")
        for h, t in ((5, "3.0s"), (4, "2.5s"), (3, "2.0s")):
            na = sum(1 for i in range(n) if P0[i][key][h] > 0)
            nb = sum(1 for i in range(n) if P1[i][key][h] > 0)
            fix = [i for i in range(n) if P0[i][key][h] > 0 and P1[i][key][h] == 0]
            brk = [i for i in range(n) if P0[i][key][h] == 0 and P1[i][key][h] > 0]
            p = binomtest(len(fix), len(fix) + len(brk), 0.5).pvalue if (fix or brk) else 1.0
            mark = "  <-- SIGNIFICANT" if p < 0.05 else ""
            print(f"    {t}: original {na:3d} -> fixed {nb:3d}   "
                  f"repaired {len(fix)}, introduced {len(brk)}   p={p:.3f}{mark}")
            res[f"{key}@{t}"] = {"orig": na, "fixed": nb,
                                 "repaired": len(fix), "introduced": len(brk), "p": float(p)}

    out = "results/quantfix_full.json" if "6019" in d0 or "full" in d0 else "results/quantfix.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    print("\nREAD IT LIKE THIS: an L2 reduction with the collision count unchanged is the")
    print("predicted outcome. An L2 reduction bought by a clear collision increase is not a")
    print("benefit and must not be reported as one.")


if __name__ == "__main__":
    main()
