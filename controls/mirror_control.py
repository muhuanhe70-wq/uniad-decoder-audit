"""Is the ego-disc arm's collision change AIMED, or would any displacement of the same size do?

THE QUESTION
------------
R-0080 (three-disc footprint) changes 18 (frame, horizon) collision cells from colliding to
clear and 8 the other way, against R-0072. Tested against zero that is p=0.167 -- underpowered,
because nuScenes val holds only 53 colliding frames. But testing against zero is also the wrong
null. The interesting question is not "did anything change" but "did the change need the
footprint geometry, or does moving the trajectory by 0.14 m in ANY direction flip cells at the
same rate?" This project has been burned by exactly that distinction before: the A2
mislocated-field arm beat the real layer on every distance metric, and the oracle lost to A2,
because those metrics reward perturbation magnitude rather than correct aiming.

THE CONTROL
-----------
For each frame take the displacement the disc arm produced,

    delta_k = T_disc[k] - T_base[k]     k = 1..6 waypoints,

decompose it at each waypoint into components parallel and perpendicular to the BASELINE
trajectory's local direction, and flip the perpendicular part:

    mirror:  delta'_k = (delta_k . u_k) u_k - (delta_k - (delta_k . u_k) u_k)

This is magnitude-matched exactly, per frame and per waypoint -- not approximately, the way an
alpha-matched rerun would be -- and it stays kinematically sensible: swerving left instead of
right is as drivable as the original manoeuvre, whereas a randomly rotated displacement would
produce sideways jumps that no vehicle could execute and that would make the null trivially
easy to beat. The longitudinal component (slow down / speed up) is preserved, so the control
isolates the one thing under test: which SIDE the layer picks.

A second control negates the whole displacement (delta'_k = -delta_k): avoid in exactly the
opposite direction, also exactly magnitude-matched.

THE TEST
--------
Per frame i, let s_true[i] = (cells repaired) - (cells introduced) for the real disc arm, and
s_ctrl[i] the same for the control arm. Under the null "direction carries no information", the
two are exchangeable within a frame, so any of the 2^n hybrid arms that pick one or the other
per frame is as likely as the observed all-true arm. Sampling those hybrids gives the null
distribution of S = sum_i s[i], and the observed S_true is located in it.

This inherits none of McNemar's dependence on the number of discordant frames being large; its
power comes from the control being matched frame by frame.

WHAT MAKES A RESULT
-------------------
  S_true >> S_ctrl   the footprint geometry aims the avoidance; the collision reduction is a
                     property of the body model, not of having perturbed the trajectory.
  S_true ~ S_ctrl    the disc arm is a perturbation generator. Same verdict the A2 and oracle
                     arms produced for the OOD layer; report as a controlled negative.
  S_true << S_ctrl   the geometry is aiming the wrong way -- a mechanism defect, and a fixable
                     one.

Collisions are recomputed offline with the dataset's own GT occupancy (dataset.prepare_test_data,
the same pipeline the live eval consumed) and the actual PlanningMetric.evaluate_coll, never a
reimplementation. The offline path is self-validated against both arms' live per_frame.json
before any control number is believed.
"""
import argparse
import gc
import io
import json
import time

import mmcv
import numpy as np
import torch
from mmcv import Config

import projects.mmdet3d_plugin  # noqa: F401
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.uniad.dense_heads.planning_head_plugin import PlanningMetric

torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")

CONFIG = "projects/configs/stage2_e2e/base_e2e.py"
NPERM = 100_000


def load_traj(path):
    b = mmcv.load(path)["bbox_results"]
    tr = np.stack([f["planning_traj"].detach().cpu().numpy()[0] for f in b]).astype(np.float64)
    del b
    gc.collect()
    return tr  # (N, 6, 2)


def local_dirs(base):
    """Unit direction of the baseline path at each of its six waypoints. (6,2)

    The ego sits at the origin at t=0, so the first segment runs from (0,0) to waypoint 1. A
    stationary step leaves the direction undefined; inherit the previous one, falling back to
    +forward, matching the convention used in collision_optimization.set_reference_trajectory.
    """
    p = np.vstack([np.zeros((1, 2)), base])
    d = np.diff(p, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    u = np.where(n > 1e-3, d / np.maximum(n, 1e-9), np.nan)
    for k in range(len(u)):
        if np.isnan(u[k]).any():
            u[k] = u[k - 1] if k > 0 and not np.isnan(u[k - 1]).any() else np.array([0.0, 1.0])
    return u


def make_controls(base, arm):
    """(mirrored, negated) trajectories with per-waypoint displacement magnitude preserved."""
    delta = arm - base
    u = local_dirs(base)
    par = (delta * u).sum(1, keepdims=True) * u          # component along the path
    perp = delta - par
    return base + par - perp, base - delta


def collide(metric, pred, gt2c, mask2, seg):
    """Per-horizon collision flags for one trajectory, via the live metric's own code path."""
    p = torch.tensor(pred, dtype=torch.float64).unsqueeze(0).clone()
    p[..., 0] = -p[..., 0]        # replicates PlanningMetric.update()'s in-place x-negation
    oc, obc = metric.evaluate_coll(p, gt2c, seg)
    return (oc.numpy() > 0), (obc.numpy() > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="work_dirs/R-0072_2026-08-17_baseline_full")
    ap.add_argument("--arm", default="work_dirs/R-0080_2026-08-19_egodiscs")
    ap.add_argument("--out", default="results/mirror_control.json")
    a = ap.parse_args()

    T0 = load_traj(a.base + "/results.pkl")
    T1 = load_traj(a.arm + "/results.pkl")
    assert T0.shape == T1.shape, "arms differ in length"
    N = len(T0)
    live0 = {r["idx"]: r for r in json.load(open(a.base + "/per_frame.json"))}
    live1 = {r["idx"]: r for r in json.load(open(a.arm + "/per_frame.json"))}

    moved = np.where(np.abs(T1 - T0).max(axis=(1, 2)) > 1e-9)[0]
    print(f"{N} frames, {len(moved)} with a non-zero displacement", flush=True)
    print("frames the arm left untouched are identical in all four arms and contribute "
          "nothing to any statistic, so only the moved frames are recomputed\n", flush=True)

    print("building dataset ...", flush=True)
    ds = build_dataset(Config.fromfile(CONFIG).data.test)
    metric = PlanningMetric()

    keys = ["base", "arm", "mirror", "neg"]
    box = {k: np.zeros((len(moved), 6), bool) for k in keys}
    pt = {k: np.zeros((len(moved), 6), bool) for k in keys}
    bad = 0
    t0 = time.time()
    for n, idx in enumerate(moved):
        d = ds.prepare_test_data(int(idx))
        gt = torch.tensor(np.array(d["sdc_planning"]).reshape(-1, 6, 3)[0:1], dtype=torch.float64)
        mask2 = torch.tensor(np.array(d["sdc_planning_mask"]).reshape(-1, 6, 2)[0:1],
                             dtype=torch.float64)
        seg = d["gt_segmentation"]
        seg = (seg[0] if isinstance(seg, list) else seg)[[1, 2, 3, 4, 5, 6]].unsqueeze(0)
        gt2c = gt[:, :, :2].clone()
        gt2c[..., 0] = -gt2c[..., 0]

        mir, neg = make_controls(T0[idx], T1[idx])
        for k, traj in zip(keys, (T0[idx], T1[idx], mir, neg)):
            pt[k][n], box[k][n] = collide(metric, traj, gt2c, mask2, seg)

        # self-validation against what the live evaluation recorded for the two real arms
        for k, live in (("base", live0), ("arm", live1)):
            if not np.array_equal(box[k][n], np.array(live[int(idx)]["obj_box_col"]) > 0):
                bad += 1
        if (n + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(moved)}  {el:.0f}s  eta {el/(n+1)*(len(moved)-n-1)/60:.1f} min",
                  flush=True)

    print(f"\nself-validation: {bad} mismatches against live per_frame.json "
          f"(out of {2*len(moved)} checks)", flush=True)
    if bad:
        raise SystemExit("offline recompute does not reproduce the live metric; nothing below "
                         "is usable")

    res = {"base": a.base.split("/")[-1], "arm": a.arm.split("/")[-1],
           "n_frames": N, "n_moved": int(len(moved))}
    rng = np.random.default_rng(0)
    for lab, tab in (("box", box), ("point", pt)):
        A = tab["base"]
        s = {}
        for k in ("arm", "mirror", "neg"):
            B = tab[k]
            s[k] = (A & ~B).sum(1).astype(int) - (~A & B).sum(1).astype(int)
            print(f"  {lab:5s} {k:6s}: repaired {int((A & ~B).sum()):3d}  "
                  f"introduced {int((~A & B).sum()):3d}  net {int(s[k].sum()):+4d}")
        res[lab] = {k: {"repaired": int((A & ~tab[k]).sum()),
                        "introduced": int((~A & tab[k]).sum()),
                        "net": int(s[k].sum())} for k in ("arm", "mirror", "neg")}
        for ctrl in ("mirror", "neg"):
            d = (s["arm"] - s[ctrl]).astype(float)
            d = d[d != 0]
            if len(d) == 0:
                p = 1.0
                obs = 0.0
            else:
                obs = d.sum()
                null = rng.choice([-1.0, 1.0], size=(NPERM, len(d))) @ d
                p = float((np.abs(null) >= abs(obs) - 1e-9).mean())
            print(f"    arm vs {ctrl}: net advantage {obs:+.0f} over {len(d)} informative "
                  f"frames, randomisation p={p:.4f}")
            res[lab][f"vs_{ctrl}"] = {"advantage": float(obs), "informative_frames": int(len(d)),
                                      "p": p}

    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
