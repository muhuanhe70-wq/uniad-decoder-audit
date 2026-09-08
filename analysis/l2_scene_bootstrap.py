"""Scene-clustered bootstrap for the paired L2 difference.

The released analyse_quantfix.py bootstraps by resampling FRAMES i.i.d.:

    b = np.array([x[r.integers(0, len(x), len(x))].mean() for _ in range(n)])

nuScenes val is 6,019 keyframes drawn from 150 scenes at 2 Hz, so frames within a
scene are strongly dependent and an i.i.d. frame bootstrap understates the
interval. The paper already clusters by frame for the collision endpoint
("cells within a frame are not independent") -- the same argument applies one
level up.

This recomputes both intervals side by side so the effect of the choice is
visible rather than asserted.

Frame order: NuScenesE2EDataset sorts infos by timestamp, so we reproduce that
ordering before indexing by `idx`.
"""
import json
import pickle
import sys

import numpy as np

HOR = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
ANN = "data/infos/nuscenes_infos_temporal_val.pkl"
A0 = "work_dirs/R-0072_2026-08-17_baseline_full/per_frame.json"
A1 = "work_dirs/R-0073_2026-08-17_quantfix_full/per_frame.json"
NBOOT = 10000


def scene_of_index():
    with open(ANN, "rb") as f:
        d = pickle.load(f)
    infos = sorted(d["infos"], key=lambda e: e["timestamp"])
    return [e["scene_token"] for e in infos]


def boot_iid(x, n=NBOOT, seed=0):
    r = np.random.default_rng(seed)
    b = np.array([x[r.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return tuple(np.percentile(b, [2.5, 97.5]))


def boot_cluster(x, groups, n=NBOOT, seed=0):
    """Resample whole scenes with replacement, then average the frames they carry.

    Each draw takes len(groups) scenes, so the resampled sample has the same
    expected number of frames as the original; scenes differ in length, which is
    exactly the variability the i.i.d. version discards.
    """
    r = np.random.default_rng(seed)
    k = len(groups)
    sums = np.array([x[g].sum() for g in groups])
    cnts = np.array([len(g) for g in groups], float)
    out = np.empty(n)
    for i in range(n):
        j = r.integers(0, k, k)
        out[i] = sums[j].sum() / cnts[j].sum()
    return tuple(np.percentile(out, [2.5, 97.5]))


def main():
    scenes = scene_of_index()
    p0 = {r["idx"]: r for r in json.load(open(A0))}
    p1 = {r["idx"]: r for r in json.load(open(A1))}
    n = len(p0)
    assert n == len(p1) == len(scenes), (n, len(p1), len(scenes))

    order = {}
    for i in range(n):
        order.setdefault(scenes[i], []).append(i)
    groups = [np.array(v) for v in order.values()]
    sizes = np.array([len(g) for g in groups])
    print(f"{n} frames in {len(groups)} scenes "
          f"(min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()})\n")

    print(f"{'horizon':>8}{'delta':>10}   {'frame iid CI':>22} {'scene-clustered CI':>24}"
          f"{'width x':>9}  ICC")
    res = {}
    for h, t in enumerate(HOR):
        a = np.array([p0[i]["L2"][h] for i in range(n)], float)
        b = np.array([p1[i]["L2"][h] for i in range(n)], float)
        dl = b - a

        lo_i, hi_i = boot_iid(dl)
        lo_c, hi_c = boot_cluster(dl, groups)
        ratio = (hi_c - lo_c) / (hi_i - lo_i)

        # one-way ICC of the paired difference across scenes
        gm = dl.mean()
        ms_b = sum(len(g) * (dl[g].mean() - gm) ** 2 for g in groups) / (len(groups) - 1)
        ms_w = sum(((dl[g] - dl[g].mean()) ** 2).sum() for g in groups) / (n - len(groups))
        m0 = (n - (sizes ** 2).sum() / n) / (len(groups) - 1)
        icc = max(0.0, (ms_b - ms_w) / (ms_b + (m0 - 1) * ms_w))

        keep = "excludes 0" if hi_c < 0 or lo_c > 0 else "INCLUDES 0"
        print(f"{t:7.1f}s{dl.mean():+10.4f}   [{lo_i:+.4f},{hi_i:+.4f}] "
              f"  [{lo_c:+.4f},{hi_c:+.4f}]{ratio:8.2f}  {icc:.3f}  {keep}")
        res[f"L2@{t}"] = {
            "delta": float(dl.mean()),
            "ci_frame_iid": [float(lo_i), float(hi_i)],
            "ci_scene_clustered": [float(lo_c), float(hi_c)],
            "width_ratio": float(ratio),
            "icc_across_scenes": float(icc),
            "excludes_zero_clustered": bool(hi_c < 0 or lo_c > 0),
        }

    res["_meta"] = {"n_frames": n, "n_scenes": len(groups), "n_boot": NBOOT, "seed": 0,
                    "note": "scene-clustered bootstrap resamples whole scenes with "
                            "replacement; the frame i.i.d. column reproduces the "
                            "released analyse_quantfix.py"}
    out = "results/l2_scene_bootstrap.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    allkeep = all(v["excludes_zero_clustered"] for k, v in res.items() if k.startswith("L2@"))
    print("all six intervals exclude zero under scene clustering:", allkeep)


if __name__ == "__main__":
    sys.exit(main())
