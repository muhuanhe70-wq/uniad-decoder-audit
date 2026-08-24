"""Does approximating the ego footprint by discs, rather than a point, reduce collisions?

THE DEFECT UNDER TEST
---------------------
`collision_optimization.py` evaluates the collision cost at the trajectory point alone, while
the benchmark scores a 4.084 x 1.85 m box offset +0.5 m forward. The front bumper therefore
sits 2.54 m from the point the optimiser reasons about, and with sigma = 1 m an obstacle
touching that bumper contributes exp(-2.54^2/2) = 3.95 % of the cost's peak -- less than the
reference-tracking term charges for a 0.28 m detour. The optimiser is close to indifferent to
a contact it is about to be penalised for.

R-0080 replaces the single sample point by three discs at longitudinal offsets
(-1.0, 0.5, 2.0) m along the reference heading, with the per-disc cost divided by three so
total cost weight is unchanged. The arm therefore isolates footprint GEOMETRY and does not
confound it with a larger alpha_collision.

Neither arm loads the OOD layer (enable_ood_injection=False in exp_egodiscs.py, and R-0072 is
the published baseline), so this measures UniAD against UniAD on identical frames.

PRE-REGISTERED CRITERION (fixed 2026-08-19, before the run finished)
--------------------------------------------------------------------
Success requires McNemar to show >= 8 collision frames repaired with <= 1 introduced. L2 is
expected to WORSEN -- pushing the body away from obstacles necessarily deviates further from
the human trajectory -- and is reported as the cost of the trade, not as a failure.

A net count that falls while repaired/introduced are both large is a reshuffle, not a fix;
that is exactly what the paired test exists to expose.
"""
import json
import sys

import numpy as np
from scipy.stats import binomtest

from analyse_quantfix import HOR, boot, load


def main():
    pat0 = sys.argv[1] if len(sys.argv) > 1 else "work_dirs/R-0072_*_baseline_full"
    pat1 = sys.argv[2] if len(sys.argv) > 2 else "work_dirs/R-0080_*_egodiscs"
    out = sys.argv[3] if len(sys.argv) > 3 else "results/egodiscs.json"
    d0, T0, ord0, P0 = load(pat0)
    d1, T1, ord1, P1 = load(pat1)
    print(f"point model (published) : {d0.split('/')[-1]}")
    print(f"three-disc footprint    : {d1.split('/')[-1]}")
    assert ord0 == ord1, "frame order differs; the runs are not comparable"
    n = len(ord0)
    print(f"paired on {n} identical frames, neither arm loads the OOD layer\n")

    diff = np.array([np.abs(T1[t] - T0[t]).max() for t in ord0])
    print(f"trajectories changed on {100*(diff > 1e-9).mean():.1f}% of frames "
          f"(mean max-abs change {diff.mean():.4f} m, p90 {np.percentile(diff,90):.4f} m)\n")

    res = {"baseline": d0.split("/")[-1], "arm": d1.split("/")[-1], "n": n}

    print("L2 to the human trajectory  (positive delta = discs deviate further -- expected)")
    print(f"{'horizon':>9}{'point':>11}{'discs':>11}{'delta':>11}{'95% CI':>22}")
    for h, t in enumerate(HOR):
        a = np.array([P0[i]["L2"][h] for i in range(n)])
        b = np.array([P1[i]["L2"][h] for i in range(n)])
        dl = b - a
        lo, hi = boot(dl)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"{t:8.1f}s{a.mean():11.4f}{b.mean():11.4f}{dl.mean():+11.4f}"
              f"   [{lo:+.4f},{hi:+.4f}]{sig}")
        res[f"L2@{t}"] = {"point": float(a.mean()), "discs": float(b.mean()),
                          "delta": float(dl.mean()), "ci": [float(lo), float(hi)],
                          "significant": bool(lo > 0 or hi < 0)}

    print("\ncollision rate  (McNemar exact on discordant pairs)")
    print("  criterion: >= 8 repaired AND <= 1 introduced")
    for key, lab in (("obj_box_col", "box"), ("obj_col", "point")):
        print(f"  {lab}:")
        for h, t in enumerate(HOR):
            na = sum(1 for i in range(n) if P0[i][key][h] > 0)
            nb = sum(1 for i in range(n) if P1[i][key][h] > 0)
            fix = sum(1 for i in range(n) if P0[i][key][h] > 0 and P1[i][key][h] == 0)
            brk = sum(1 for i in range(n) if P0[i][key][h] == 0 and P1[i][key][h] > 0)
            p = binomtest(fix, fix + brk, 0.5).pvalue if (fix + brk) else 1.0
            mark = "  <-- p<0.05" if p < 0.05 else ""
            meets = " [MEETS CRITERION]" if (fix >= 8 and brk <= 1) else ""
            print(f"    {t:.1f}s: point {na:3d} -> discs {nb:3d}   "
                  f"repaired {fix:3d}, introduced {brk:3d}   p={p:.3f}{mark}{meets}")
            res[f"{key}@{t}"] = {"point": na, "discs": nb, "repaired": fix,
                                 "introduced": brk, "p": float(p)}

        # Frames colliding at ANY horizon: the union is what a deployed system cares about,
        # and it has more discordant pairs than any single horizon, so it is the test with
        # the most power. Reported alongside, never instead of, the per-horizon numbers.
        A = np.array([any(P0[i][key][h] > 0 for h in range(len(HOR))) for i in range(n)])
        B = np.array([any(P1[i][key][h] > 0 for h in range(len(HOR))) for i in range(n)])
        fix, brk = int((A & ~B).sum()), int((~A & B).sum())
        p = binomtest(fix, fix + brk, 0.5).pvalue if (fix + brk) else 1.0
        meets = " [MEETS CRITERION]" if (fix >= 8 and brk <= 1) else ""
        print(f"    any  : point {int(A.sum()):3d} -> discs {int(B.sum()):3d}   "
              f"repaired {fix:3d}, introduced {brk:3d}   p={p:.3f}{meets}")
        res[f"{key}@any"] = {"point": int(A.sum()), "discs": int(B.sum()),
                             "repaired": fix, "introduced": brk, "p": float(p)}

    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
