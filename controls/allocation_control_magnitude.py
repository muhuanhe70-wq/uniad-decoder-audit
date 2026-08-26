"""Allocation control, magnitude-matched: was the avoidance withdrawn on the RIGHT FRAMES?

WHY THIS EXISTS
---------------
`allocation_control.py` permuted the per-frame retention factor s_i and was reported as
holding "the aggregate withdrawal" fixed. It does not (see
analysis/audit_allocation_magnitude.py): permuting s preserves the multiset of retention
FACTORS, but the quantity the objection concerns is the applied displacement

    m_i = s_i * ||A^B_i||

and s correlates with ||A^B|| at r = +0.468, so the permuted arms apply only 67.3% of the
reference's total avoidance. Every unit of that control's apparent advantage arose on
frames where the permuted arm avoided less, so it could not separate "withdrew on the
right frames" from "withdrew less in total". Its conclusion is withdrawn in the paper.

WHAT THIS DOES INSTEAD
----------------------
Permute the applied MAGNITUDES m_i rather than the factors, holding each frame's
published avoidance DIRECTION fixed:

    reference   Fhat_i = R_i + s_i                        * A^B_i
    control     Ftil_i = R_i + (m_pi(i) / ||A^B_i||)      * A^B_i

Then ||Ftil_i - R_i|| = m_pi(i) exactly, so the multiset {m_i} -- and hence the total
sum(m_i), and the maximum, and every other order statistic -- is reproduced exactly. Only
which frame receives which magnitude changes.

THE THRESHOLD, AND WHY IT IS NOT OPTIONAL
-----------------------------------------
||A^B_i|| spans five orders of magnitude, from 3e-6 m to 2.5 m. On a frame where the
published optimiser applied 50 micrometres of avoidance, the *direction* A^B_i/||A^B_i||
is numerical noise, not a bearing. Handing such a frame a half-metre magnitude would
apply a large displacement along a meaningless direction -- a random-direction
perturbation dressed up as a reallocation, and a strawman this project's own controls
exist to avoid. Measured: an unrestricted permutation gives scale factors above 10 on
20% of (frame, draw) pairs and a 99th percentile of 2e4.

So the permutation runs only among frames with ||A^B_i|| >= TAU = 0.10 m. That threshold
is not tuned: 0.10 m is the displacement at which Section 6.2 shows the benchmark's
collision count already responds (an arbitrary 0.10 m move introduces 38 new colliding
cells), so below it a frame's avoidance direction cannot affect the scored outcome
either way. It retains 1,681 of the 3,338 eligible frames and 97.8% of the total applied
avoidance, and bounds the scale factor at 11.7 with the largest displacement equal to
the reference's own maximum, 1.47 m.

Frames below TAU are held at the reference value in both arms, so they contribute
identically and cancel; only frames in the permuted set are scored.

READING IT
----------
If Fhat repairs collisions significantly better than the permuted allocations, the
avoidance was withdrawn where it was spurious. If it does not, the diminishing-returns
account stands and the paper says so. This is a real test and it can go against us.
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
EPS = 1e-6
TAU = 0.10
NPERM = 100_000


def load_traj(path):
    b = mmcv.load(path)["bbox_results"]
    tr = np.stack([f["planning_traj"].detach().cpu().numpy()[0]
                   for f in b]).astype(np.float64)
    del b
    gc.collect()
    return tr


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
    ap.add_argument("--out", default="results/allocation_control_magnitude.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=TAU)
    a = ap.parse_args()

    ckpt = a.out + ".partial.npz"

    B = load_traj(a.base + "/results.pkl")
    F = load_traj(a.arm + "/results.pkl")
    R = load_traj(a.raw + "/results.pkl")
    assert B.shape == F.shape == R.shape, "arms differ in length"
    live0 = {r["idx"]: r for r in json.load(open(a.base + "/per_frame.json"))}
    live1 = {r["idx"]: r for r in json.load(open(a.arm + "/per_frame.json"))}

    AB = B - R
    AF = F - R
    nB = np.linalg.norm(AB, axis=2).mean(1)
    nF = np.linalg.norm(AF, axis=2).mean(1)

    moved_all = np.where(np.abs(F - B).max(axis=(1, 2)) > 1e-9)[0]
    eligible = nB[moved_all] > EPS
    moved = moved_all[eligible]
    s = nF[moved] / nB[moved]
    m = s * nB[moved]                       # == nF[moved], the applied magnitude

    keep = nB[moved] >= a.tau               # frames whose direction is meaningful
    idxs = moved[keep]
    mk, nBk = m[keep], nB[moved][keep]
    K = a.k

    print("%d frames moved, %d eligible, %d above tau=%.2f m (%.1f%% of the applied "
          "avoidance)" % (len(moved_all), len(moved), len(idxs), a.tau,
                          100 * mk.sum() / m.sum()), flush=True)

    rng = np.random.default_rng(a.seed)
    perms = np.stack([rng.permutation(len(idxs)) for _ in range(K)])
    scale = mk[perms] / nBk                 # (K, n) scale factor per draw and frame
    print("scale factor: median %.2f, max %.2f; displacement max %.3f m against the "
          "reference's %.3f m" % (np.median(scale), scale.max(),
                                  mk[perms].max(), m.max()), flush=True)

    # box[0] baseline, box[1] real arm F, box[2] Fhat, box[3:] the K permuted allocations
    box = np.zeros((3 + K, len(idxs), 6), bool)
    start = 0
    if os.path.exists(ckpt):
        d = np.load(ckpt)
        if d["perms"].shape == perms.shape and int(d["done"]) > 0:
            box, perms, start = d["box"], d["perms"], int(d["done"])
            print("resuming from checkpoint at frame %d" % start, flush=True)

    print("building dataset ...", flush=True)
    ds = build_dataset(Config.fromfile(CONFIG).data.test)
    metric = PlanningMetric()

    bad = 0
    t0 = time.time()
    for n in range(start, len(idxs)):
        idx = int(idxs[n])
        d = ds.prepare_test_data(idx)
        gt = torch.tensor(np.array(d["sdc_planning"]).reshape(-1, 6, 3)[0:1],
                          dtype=torch.float64)
        mask2 = torch.tensor(np.array(d["sdc_planning_mask"]).reshape(-1, 6, 2)[0:1],
                             dtype=torch.float64)
        seg = d["gt_segmentation"]
        seg = (seg[0] if isinstance(seg, list) else seg)[[1, 2, 3, 4, 5, 6]].unsqueeze(0)
        gt2c = gt[:, :, :2].clone()
        gt2c[..., 0] = -gt2c[..., 0]

        j = int(np.where(moved == idx)[0][0])
        variants = [B[idx], F[idx], R[idx] + s[j] * AB[idx]]
        variants += [R[idx] + scale[q][n] * AB[idx] for q in range(K)]
        for v, traj in enumerate(variants):
            _, box[v][n] = collide(metric, traj, gt2c, mask2, seg)

        for v, live in ((0, live0), (1, live1)):
            if not np.array_equal(box[v][n], np.array(live[idx]["obj_box_col"]) > 0):
                bad += 1

        if (n + 1) % 100 == 0:
            el = time.time() - t0
            rate = el / (n + 1 - start)
            print("  %d/%d  %.0fs  eta %.1f min"
                  % (n + 1, len(idxs), el, rate * (len(idxs) - n - 1) / 60), flush=True)
            np.savez_compressed(ckpt, box=box, perms=perms, done=n + 1)

    print("\nself-validation: %d mismatches against live per_frame.json (out of %d)"
          % (bad, 2 * (len(idxs) - start)), flush=True)

    base = box[0]

    def net(v):
        return ((base & ~box[v]).sum(1).astype(float)
                - (~base & box[v]).sum(1).astype(float))

    real_arm = net(1)
    real_alloc = net(2)
    perm = np.stack([net(3 + q) for q in range(K)])

    print("\nreal arm F (reference)        : %+.0f" % real_arm.sum())
    print("real allocation, pub direction: %+.0f" % real_alloc.sum())
    print("permuted allocation, mean     : %+.1f  (min %+.0f, max %+.0f over %d draws)"
          % (perm.sum(1).mean(), perm.sum(1).min(), perm.sum(1).max(), K))
    print("permutations beating real     : %d/%d"
          % ((perm.sum(1) >= real_alloc.sum()).sum(), K))

    diff = real_alloc - perm.mean(0)
    obs = diff.sum()
    inf = int((diff != 0).sum())
    rng2 = np.random.default_rng(0)
    cnt = 0
    for _ in range(NPERM):
        sg = rng2.choice([-1.0, 1.0], size=diff.shape)
        if (diff * sg).sum() >= obs:
            cnt += 1
    p = (cnt + 1) / (NPERM + 1)
    print("advantage %+.1f over %d informative frames, sign-permutation p = %.5f"
          % (obs, inf, p))

    out = {
        "base": os.path.basename(a.base), "arm": os.path.basename(a.arm),
        "raw": os.path.basename(a.raw), "k": K, "seed": a.seed, "tau": a.tau,
        "n_moved": int(len(moved_all)), "n_eligible": int(len(moved)),
        "n_permuted": int(len(idxs)),
        "share_of_applied_avoidance_permuted": float(mk.sum() / m.sum()),
        "total_applied_reference": float(m.sum()),
        "total_applied_permuted": float(mk[perms[0]].sum() + m[~keep].sum()),
        "max_scale_factor": float(scale.max()),
        "self_validation_mismatches": int(bad),
        "real_arm_net": float(real_arm.sum()),
        "real_allocation_net": float(real_alloc.sum()),
        "permuted_net_per_draw": perm.sum(1).tolist(),
        "permutations_beating_real": int((perm.sum(1) >= real_alloc.sum()).sum()),
        "vs_mean_permuted": {"advantage": float(obs), "informative_frames": inf,
                             "p": float(p)},
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    np.savez_compressed(a.out.replace(".json", "_box.npz"),
                        box=box, perms=perms, idxs=idxs, scale=scale,
                        real_alloc=real_alloc, perm=perm)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
