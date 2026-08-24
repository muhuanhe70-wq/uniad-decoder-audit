"""Uniform-random-direction control: the strong version of the mirror control.

WHY THIS EXISTS
---------------
`mirror_control.py` holds the per-waypoint displacement magnitude fixed and flips its
cross-path component. That is exactly magnitude-matched, but it does not sample directions --
it produces one specific alternative direction, and how informative that is depends on where
that direction happens to land. Measured on 2026-08-21, none of our three mirrors is
orthogonal: the median direction cosine between the real and mirrored displacement is -0.512
for the coordinate fix, -0.216 for the mean-reduction discs and -0.768 for the summed discs.
A mirror that leans anti-parallel is close to a negation control, and a negation control is
the weakest possible direction control -- a reviewer can dismiss it as a strawman.

This script removes that objection. For each moved frame it draws K angles uniformly from
[0, 2*pi) and rotates the frame's whole displacement field by each of them:

    b + R(theta) * delta ,    theta ~ U[0, 2*pi)

R is a rotation, so ||R(theta) delta_t|| == ||delta_t|| for every waypoint: the magnitude
match is exact by construction, not approximate. One angle is used per frame (not per
waypoint) so the shape of the avoidance manoeuvre is preserved and only its bearing changes,
which keeps the control kinematically comparable to the real arm. theta = 0 reproduces the
arm and theta = pi reproduces the negation control, so the mirror and negation results are
interior points of the distribution this script samples.

WHAT IT REPORTS
---------------
Two readings, both frame-paired:

1. Rank test. For each frame, the real displacement's net collision contribution
   (repaired - introduced, over the six horizons) is compared against the same quantity for
   each of the K rotations. Reporting the fraction of (frame, rotation) pairs the real
   direction loses to gives a direct randomisation p-value for "this direction is no better
   than an arbitrary one of the same size".

2. Frame-clustered sign-permutation test on the paired difference between the real net and
   the frame's mean rotated net, matching the test used everywhere else in this project.

COST
----
The expensive step is decoding a frame through the dataset pipeline, which is done once per
frame and shared across all K rotations, so K is close to free relative to a second full pass.
Partial results are checkpointed so an interrupted run is not lost.

Collisions are recomputed with the benchmark's own PlanningMetric.evaluate_coll on the
dataset's own ground truth, never a reimplementation, and the real arm's recomputed flags are
self-validated against the live per_frame.json before any control number is believed.
"""
import argparse
import gc
import io
import json
import os
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


def rotate(delta, theta):
    """Rotate a (6,2) displacement field by a single angle. Magnitude is preserved exactly."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return delta @ R.T


def collide(metric, pred, gt2c, mask2, seg):
    """Replicates PlanningMetric.update()'s in-place x-negation, then scores."""
    p = torch.tensor(pred, dtype=torch.float64).unsqueeze(0).clone()
    p[..., 0] = -p[..., 0]
    oc, obc = metric.evaluate_coll(p, gt2c, seg)
    return (oc.numpy() > 0), (obc.numpy() > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="work_dirs/R-0072_2026-08-17_baseline_full")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8, help="random directions per frame")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ckpt = a.out + ".partial.npz"

    T0 = load_traj(a.base + "/results.pkl")
    T1 = load_traj(a.arm + "/results.pkl")
    assert T0.shape == T1.shape, "arms differ in length"
    live0 = {r["idx"]: r for r in json.load(open(a.base + "/per_frame.json"))}
    live1 = {r["idx"]: r for r in json.load(open(a.arm + "/per_frame.json"))}

    moved = np.where(np.abs(T1 - T0).max(axis=(1, 2)) > 1e-9)[0]
    K = a.k
    print(f"{len(T0)} frames, {len(moved)} moved, {K} random directions each", flush=True)

    rng = np.random.default_rng(a.seed)
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=(len(moved), K))

    # box[0] = baseline, box[1] = real arm, box[2:] = the K rotations
    box = np.zeros((2 + K, len(moved), 6), bool)
    start = 0
    if os.path.exists(ckpt):
        d = np.load(ckpt)
        if d["thetas"].shape == thetas.shape and int(d["done"]) > 0:
            box, thetas, start = d["box"], d["thetas"], int(d["done"])
            print(f"resuming from checkpoint at frame {start}", flush=True)

    print("building dataset ...", flush=True)
    ds = build_dataset(Config.fromfile(CONFIG).data.test)
    metric = PlanningMetric()

    bad = 0
    t0 = time.time()
    for n in range(start, len(moved)):
        idx = int(moved[n])
        d = ds.prepare_test_data(idx)
        gt = torch.tensor(np.array(d["sdc_planning"]).reshape(-1, 6, 3)[0:1], dtype=torch.float64)
        mask2 = torch.tensor(np.array(d["sdc_planning_mask"]).reshape(-1, 6, 2)[0:1],
                             dtype=torch.float64)
        seg = d["gt_segmentation"]
        seg = (seg[0] if isinstance(seg, list) else seg)[[1, 2, 3, 4, 5, 6]].unsqueeze(0)
        gt2c = gt[:, :, :2].clone()
        gt2c[..., 0] = -gt2c[..., 0]

        delta = T1[idx] - T0[idx]
        variants = [T0[idx], T1[idx]] + [T0[idx] + rotate(delta, th) for th in thetas[n]]
        for v, traj in enumerate(variants):
            _, box[v][n] = collide(metric, traj, gt2c, mask2, seg)

        for v, live in ((0, live0), (1, live1)):
            if not np.array_equal(box[v][n], np.array(live[idx]["obj_box_col"]) > 0):
                bad += 1

        if (n + 1) % 200 == 0:
            el = time.time() - t0
            rate = el / (n + 1 - start)
            print(f"  {n+1}/{len(moved)}  {el:.0f}s  eta {rate*(len(moved)-n-1)/60:.1f} min",
                  flush=True)
            np.savez_compressed(ckpt, box=box, thetas=thetas, done=n + 1)

    print(f"\nself-validation: {bad} mismatches against live per_frame.json "
          f"(out of {2*(len(moved)-start)} checks)", flush=True)

    # ---- per-frame net collision contribution: repaired minus introduced ----
    base = box[0]
    def net(v):
        return (base & ~box[v]).sum(1).astype(float) - (~base & box[v]).sum(1).astype(float)

    real = net(1)
    rot = np.stack([net(2 + j) for j in range(K)])            # (K, n_moved)

    # 1) direct randomisation: how often does an arbitrary direction match or beat the real one
    wins = (rot >= real[None, :]).sum()
    total = rot.size
    print(f"\nreal net total       : {real.sum():+.0f}")
    print(f"rotated net, mean    : {rot.sum(1).mean():+.1f}  "
          f"(min {rot.sum(1).min():+.0f}, max {rot.sum(1).max():+.0f} over {K} draws)")
    print(f"draws beating real   : {(rot.sum(1) >= real.sum()).sum()}/{K}")
    print(f"(frame,draw) pairs where an arbitrary direction is at least as good: "
          f"{wins}/{total} = {wins/total:.4f}")

    # The line above is dominated by ties and must never be quoted on its own: on most changed
    # frames neither arm collides, so net==0 for both and ">=" scores the tie as "at least as
    # good". Decompose it, so the paper can state the tie rate alongside it instead of leaving
    # a reader to assume the 0.99 means an arbitrary direction is as good as the real one.
    ties = int((rot == real[None, :]).sum())
    strict_wins = int((rot > real[None, :]).sum())
    losses = int((rot < real[None, :]).sum())
    both_zero = int(((rot == 0) & (real[None, :] == 0)).sum())
    print(f"  decomposed: {ties} ties ({both_zero} of them net==0 on both), "
          f"{strict_wins} strict wins for an arbitrary direction, {losses} losses")
    nz = real != 0
    if nz.any():
        rz, realz = rot[:, nz], real[None, nz]
        print(f"  restricted to the {int(nz.sum())} frames the real fix changes the outcome on: "
              f"arbitrary direction strictly better in {int((rz > realz).sum())}/{rz.size}, "
              f"tied in {int((rz == realz).sum())}, worse in {int((rz < realz).sum())}")

    # 2) frame-clustered sign permutation on real minus mean-rotated
    diff = real - rot.mean(0)
    inf = diff[np.abs(diff) > 1e-12]
    obs = inf.sum()
    if len(inf):
        r = np.random.default_rng(1)
        signs = r.integers(0, 2, size=(NPERM, len(inf))) * 2 - 1
        null = (signs * np.abs(inf)).sum(1)
        p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    else:
        p = 1.0
    print(f"\nreal vs mean-rotated : net advantage {obs:+.1f} over {len(inf)} informative "
          f"frames, frame-clustered permutation p={p:.4f}")

    res = {
        "base": a.base.split("/")[-1], "arm": a.arm.split("/")[-1],
        "k": K, "seed": a.seed, "n_moved": int(len(moved)),
        "self_validation_mismatches": int(bad),
        "real_net": float(real.sum()),
        "rotated_net_per_draw": [float(x) for x in rot.sum(1)],
        "draws_beating_real": int((rot.sum(1) >= real.sum()).sum()),
        "pairwise_frac_arbitrary_at_least_as_good": float(wins / total),
        "pairwise_decomposition": {
            "ties": ties, "ties_both_zero": both_zero,
            "strict_wins_for_arbitrary": strict_wins, "losses_for_arbitrary": losses,
            "frames_real_changes_outcome": int(nz.sum()),
            "restricted_strictly_better": int((rot[:, nz] > real[None, nz]).sum()) if nz.any() else 0,
            "restricted_tied": int((rot[:, nz] == real[None, nz]).sum()) if nz.any() else 0,
            "restricted_worse": int((rot[:, nz] < real[None, nz]).sum()) if nz.any() else 0,
        },
        "vs_mean_rotated": {"advantage": float(obs), "informative_frames": int(len(inf)),
                            "p": p},
    }
    json.dump(res, open(a.out, "w"), indent=2)
    # Keep the scored outcomes. Deleting them on success cost a re-run once already: the
    # tie decomposition above could not be computed after the fact because `box` was gone.
    np.savez_compressed(a.out.replace(".json", "_box.npz"), box=box, thetas=thetas,
                        moved=moved, real=real, rot=rot)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"\nwrote {a.out} and {a.out.replace('.json', '_box.npz')}")


if __name__ == "__main__":
    main()
