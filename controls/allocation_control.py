"""Allocation control: was the withdrawn avoidance withdrawn on the RIGHT FRAMES?

WHY THIS EXISTS
---------------
Every control in this project so far -- the mirror, the negation, the uniform random rotation
-- holds the displacement magnitude fixed and varies direction. That is what makes them clean
tests of aiming, and it is exactly what they cannot speak to, because the R-0084 measurement
(analyse_avoidance.py, 2026-08-22) showed that magnitude is what predominantly changed:

    avoidance applied by the optimiser   0.2965 m  ->  0.1582 m   (-47%)
    frames avoiding MORE after the fix   1.0%
    of the change in the planned path    94% magnitude, 6% direction

So the objection "you avoided less rather than planning better" is factually correct about
what the correction does, and our direction controls hold fixed precisely the quantity that
moved. The three-arm table (published / corrected / optimiser off) shows collisions are NOT
monotone in avoidance -- the corrected arm has the fewest of the three -- but that is also
consistent with a purely magnitude-based account in which avoidance has diminishing returns,
so that keeping half of it keeps most of the benefit no matter which frames it is kept on.

This script separates those. It holds the aggregate withdrawal fixed and permutes only its
ALLOCATION across frames.

CONSTRUCTION
------------
Let B be the published trajectory, F the corrected one and R the raw network trajectory with
the collision optimiser disabled (R-0084). Then A^B = B - R and A^F = F - R are the avoidance
each arm's optimiser applied. Per frame define the retention factor

    s_i = ||A^F_i|| / ||A^B_i||          (mean over the six waypoints)

which is how much of the published avoidance the correction kept on that frame. Build

    reference   Fhat_i = R_i + s_i        * A^B_i     real allocation, published direction
    control     Ftil_i = R_i + s_{pi(i)}  * A^B_i     permuted allocation, published direction

Both keep every frame's avoidance DIRECTION exactly as published, and both draw their scale
factors from the identical multiset {s_i}. The only difference is which frame received which
factor.

*** CORRECTION, 2026-08-25: the last step of that reasoning is wrong. ***

Permuting {s_i} preserves the multiset of retention FACTORS, but the quantity the objection is
about is the applied displacement m_i = s_i * ||A^B_i||, and permuting s while leaving
||A^B_i|| where it is preserves sum(m) only if s and ||A^B|| are uncorrelated. They are not:
r = +0.468. The permuted allocations therefore apply only 67.3% of the reference's total
avoidance, and analysis/audit_allocation_magnitude.py shows that 100% of the advantage this
script reports arises on the frames where the permuted allocation applied LESS. This control
does NOT separate allocation from magnitude and its result is withdrawn in the paper.

The control that would settle it permutes the applied MAGNITUDES rather than the factors:

    Ftil_i = R_i + (m_{pi(i)} / ||A^B_i||) * A^B_i

which reproduces the multiset {m_i}, and hence sum(m), exactly. That variant HAS since been run
and is what the paper now reports: see controls/allocation_control_magnitude.py and
results/allocation_control_magnitude.json (the two arms' totals agree to 1e-13 m; the real
allocation nets +12 against -15.0, 0/8 draws better, p = 6e-5).

This file is left exactly as it was run, so that the withdrawn numbers remain reproducible and
the audit above can be checked against them.

Note that Fhat is deliberately NOT the real arm F: stripping the 6% directional change removes
the advantage the rotation control has already established, so that this test measures the one
thing it is meant to measure. F is scored anyway, as variant 1, for reference.

READING IT
----------
If Fhat repairs collisions significantly better than the permuted allocations, the avoidance
was withdrawn where it was spurious and the interpretation in the paper holds. If it does not,
the diminishing-returns account stands, and Section 7 must weaken accordingly. This is a real
test and it can go against us; do not pick the favourable reading afterwards.

Frames where the published optimiser applied essentially no avoidance (||A^B_i|| <= EPS) have
no retention factor to permute and are excluded; their count is reported.
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
EPS = 1e-6


def load_traj(path):
    b = mmcv.load(path)["bbox_results"]
    tr = np.stack([f["planning_traj"].detach().cpu().numpy()[0] for f in b]).astype(np.float64)
    del b
    gc.collect()
    return tr  # (N, 6, 2)


def collide(metric, pred, gt2c, mask2, seg):
    """Replicates PlanningMetric.update()'s in-place x-negation, then scores."""
    p = torch.tensor(pred, dtype=torch.float64).unsqueeze(0).clone()
    p[..., 0] = -p[..., 0]
    oc, obc = metric.evaluate_coll(p, gt2c, seg)
    return (oc.numpy() > 0), (obc.numpy() > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="work_dirs/R-0072_2026-08-17_baseline_full")
    ap.add_argument("--arm", default="work_dirs/R-0073_2026-08-17_quantfix_full")
    ap.add_argument("--raw", default="work_dirs/R-0084_2026-08-21_nocoloptim")
    ap.add_argument("--out", default="results/allocation_control.json")
    ap.add_argument("--k", type=int, default=8, help="permutations of the allocation")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ckpt = a.out + ".partial.npz"

    B = load_traj(a.base + "/results.pkl")
    F = load_traj(a.arm + "/results.pkl")
    R = load_traj(a.raw + "/results.pkl")
    assert B.shape == F.shape == R.shape, "arms differ in length"
    live0 = {r["idx"]: r for r in json.load(open(a.base + "/per_frame.json"))}
    live1 = {r["idx"]: r for r in json.load(open(a.arm + "/per_frame.json"))}

    AB = B - R                                        # (N, 6, 2) published avoidance
    AF = F - R                                        # (N, 6, 2) corrected avoidance
    nB = np.linalg.norm(AB, axis=2).mean(1)           # (N,)
    nF = np.linalg.norm(AF, axis=2).mean(1)

    moved_all = np.where(np.abs(F - B).max(axis=(1, 2)) > 1e-9)[0]
    eligible = nB[moved_all] > EPS
    moved = moved_all[eligible]
    n_skipped = int((~eligible).sum())

    s = nF[moved] / nB[moved]                         # retention factor per eligible frame
    K = a.k
    print(f"{len(B)} frames, {len(moved_all)} moved, {len(moved)} eligible "
          f"({n_skipped} skipped: the published optimiser applied no avoidance)", flush=True)
    print(f"retention factor s: mean {s.mean():.4f}, median {np.median(s):.4f}, "
          f"min {s.min():.4f}, max {s.max():.4f}", flush=True)
    print(f"{K} permutations of the allocation, seed {a.seed}", flush=True)

    rng = np.random.default_rng(a.seed)
    perms = np.stack([rng.permutation(len(moved)) for _ in range(K)])   # (K, n_moved)

    # box[0] baseline, box[1] real arm F, box[2] Fhat, box[3:] the K permuted allocations
    box = np.zeros((3 + K, len(moved), 6), bool)
    start = 0
    if os.path.exists(ckpt):
        d = np.load(ckpt)
        if d["perms"].shape == perms.shape and int(d["done"]) > 0:
            box, perms, start = d["box"], d["perms"], int(d["done"])
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

        variants = [B[idx], F[idx], R[idx] + s[n] * AB[idx]]
        variants += [R[idx] + s[perms[j][n]] * AB[idx] for j in range(K)]
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
            np.savez_compressed(ckpt, box=box, perms=perms, done=n + 1)

    print(f"\nself-validation: {bad} mismatches against live per_frame.json "
          f"(out of {2*(len(moved)-start)} checks)", flush=True)

    base = box[0]

    def net(v):
        return (base & ~box[v]).sum(1).astype(float) - (~base & box[v]).sum(1).astype(float)

    real_arm = net(1)      # the actual corrected arm, for reference only
    real_alloc = net(2)    # real allocation, published direction  <- the thing under test
    perm = np.stack([net(3 + j) for j in range(K)])                 # (K, n_moved)

    print(f"\nreal arm F (reference)        : {real_arm.sum():+.0f}")
    print(f"real allocation, pub direction: {real_alloc.sum():+.0f}")
    print(f"permuted allocation, mean     : {perm.sum(1).mean():+.1f}  "
          f"(min {perm.sum(1).min():+.0f}, max {perm.sum(1).max():+.0f} over {K} draws)")
    print(f"permutations beating real     : {(perm.sum(1) >= real_alloc.sum()).sum()}/{K}")

    ties = int((perm == real_alloc[None, :]).sum())
    both_zero = int(((perm == 0) & (real_alloc[None, :] == 0)).sum())
    swins = int((perm > real_alloc[None, :]).sum())
    losses = int((perm < real_alloc[None, :]).sum())
    print(f"  decomposed: {ties} ties ({both_zero} of them net==0 on both), "
          f"{swins} strict wins for a permuted allocation, {losses} losses")

    diff = real_alloc - perm.mean(0)
    inf = diff[np.abs(diff) > 1e-12]
    obs = inf.sum()
    if len(inf):
        r = np.random.default_rng(1)
        signs = r.integers(0, 2, size=(NPERM, len(inf))) * 2 - 1
        null = (signs * np.abs(inf)).sum(1)
        p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    else:
        p = 1.0
    print(f"\nreal vs mean-permuted allocation: net advantage {obs:+.1f} over {len(inf)} "
          f"informative frames, frame-clustered permutation p={p:.4f}")
    print("\nREAD IT HONESTLY: a null result here means the withdrawal of avoidance was not")
    print("shown to be better allocated than chance, and Section 7 weakens accordingly.")

    res = {
        "base": a.base.split("/")[-1], "arm": a.arm.split("/")[-1],
        "raw": a.raw.split("/")[-1], "k": K, "seed": a.seed,
        "n_moved": int(len(moved_all)), "n_eligible": int(len(moved)), "n_skipped": n_skipped,
        "retention_mean": float(s.mean()), "retention_median": float(np.median(s)),
        "self_validation_mismatches": int(bad),
        "real_arm_net": float(real_arm.sum()),
        "real_allocation_net": float(real_alloc.sum()),
        "permuted_net_per_draw": [float(x) for x in perm.sum(1)],
        "permutations_beating_real": int((perm.sum(1) >= real_alloc.sum()).sum()),
        "pairwise_decomposition": {"ties": ties, "ties_both_zero": both_zero,
                                   "strict_wins_for_permuted": swins,
                                   "losses_for_permuted": losses},
        "vs_mean_permuted": {"advantage": float(obs), "informative_frames": int(len(inf)),
                             "p": p},
    }
    json.dump(res, open(a.out, "w"), indent=2)
    np.savez_compressed(a.out.replace(".json", "_box.npz"), box=box, perms=perms,
                        moved=moved, s=s, real_alloc=real_alloc, perm=perm)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
