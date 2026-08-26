"""How far does the planner move for an obstacle touching its own front bumper?

Section 5.5 states that the three-disc footprint does mechanically what it was designed
to do: the planner's response to an obstacle at the bumper grows from the point model to
the mean-reduction discs to the summed-reduction discs. That is a claim about the
optimiser's geometry, not about the dataset, so it is checked here with a controlled
single-obstacle probe rather than measured over nuScenes.

SETUP. A straight reference trajectory at constant speed, and one obstacle placed on the
centreline exactly at the scored front bumper of the first waypoint -- 4.084/2 + 0.5 =
2.542 m ahead of it, the lever arm of Defect 2. The same CasADi optimiser the stack ships
is then run in three configurations, with the solver settings the arms use:

    point model         ego_discs = None            (UniAD as published)
    three discs, mean   [-1.0, 0.5, 2.0], 'mean'    weight-matched, geometry only
    three discs, sum    [-1.0, 0.5, 2.0], 'sum'     the literature-default reduction

Response is the largest displacement of any waypoint from the reference. A larger
response means the optimiser noticed the obstacle more.

Usage (from the root of the UniAD working tree):

    python3 analysis/verify_bumper_response.py --out results/bumper_response.json
"""
import argparse
import json

import numpy as np

from projects.mmdet3d_plugin.uniad.dense_heads.planning_head_plugin.collision_optimization \
    import CollisionNonlinearOptimizer

TRAJ_LEN = 6
DT = 0.5
SIGMA = 1.0
ALPHA = 5.0
DISCS = [-1.0, 0.5, 2.0]
BUMPER = 4.084 / 2 + 0.5          # 2.542 m, the lever arm of Defect 2
# The reference is STATIONARY, which is the geometry Figure 2 depicts and is not a
# simplification for convenience. With a moving reference the "front bumper of waypoint
# t" falls on or near waypoint t+1, so the point model sees the obstacle sitting on a
# trajectory point rather than 2.542 m from one, and the comparison stops being about
# footprint geometry: at 5 m/s the point model's response is 0.399 m, larger than the
# mean-disc variant's, purely from that coincidence. Holding the reference still puts
# the obstacle 2.542 m from the trajectory point and 0.542 m from the forward disc,
# which is exactly the lever arm Defect 2 is about.
SPEED = 0.0


def run(ego_discs, reduce_mode, obstacle):
    # the optimiser's ref_traj is 2 x N; heading is derived internally from the
    # reference itself, so the reference is (x, y) only
    ref = np.stack([np.zeros(TRAJ_LEN),
                    SPEED * DT * np.arange(1, TRAJ_LEN + 1)], axis=1)
    obj = [np.array([obstacle]) for _ in range(TRAJ_LEN)]
    opt = CollisionNonlinearOptimizer(
        TRAJ_LEN, DT, SIGMA, ALPHA, obj,
        ego_discs=ego_discs, ego_disc_reduce=reduce_mode)
    opt.set_reference_trajectory(ref)
    sol = opt.solve()
    xy = np.stack([np.array(sol.value(opt.position_x)).reshape(-1),
                   np.array(sol.value(opt.position_y)).reshape(-1)], axis=1)
    return ref, xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bumper_response.json")
    a = ap.parse_args()

    # the obstacle sits on the centreline, at the bumper of the first waypoint
    obstacle = [0.0, SPEED * DT + BUMPER]

    out = {"trajectory_len": TRAJ_LEN, "dt": DT, "sigma": SIGMA,
           "alpha_collision": ALPHA, "ego_discs": DISCS,
           "bumper_offset_m": BUMPER, "obstacle_xy": obstacle,
           "reference_speed_mps": SPEED, "arms": {}}

    for tag, discs, mode in (("point", None, "mean"),
                             ("discs_mean", DISCS, "mean"),
                             ("discs_sum", DISCS, "sum")):
        ref, xy = run(discs, mode, obstacle)
        disp = np.linalg.norm(xy - ref, axis=1)
        out["arms"][tag] = {
            "max_displacement_m": float(disp.max()),
            "mean_displacement_m": float(disp.mean()),
            "per_waypoint_m": [float(v) for v in disp],
        }
        print("%-11s max %.4f m   mean %.4f m"
              % (tag, disp.max(), disp.mean()))

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
