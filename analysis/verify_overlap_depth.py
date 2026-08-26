"""How deeply does the scored ego footprint overlap occupancy, for the collisions the
coordinate correction removes versus the ones it leaves?

The paper uses this as a consistency check on the mechanism: a half-metre displacement
of the obstacle set should flip grazing contacts and leave deep overlaps alone. If it
did the opposite, the one-line change would be doing something other than what is
claimed for it. The figures were originally taken from console output; this script
re-derives them so they rest on a file.

METHOD. `PlanningMetric.evaluate_single_coll` rasterises the ego box into 32 grid cells
and returns `np.any(occupied)`. This reuses that computation verbatim -- same `bx`,
`dx`, `H`, `W`, same `skimage.draw.polygon` call -- and changes only the reduction, from
`any` to a count, so the depth reported is the depth the benchmark itself tests.

Two conventions from the benchmark are preserved deliberately, because reimplementing
around either produces numbers that look impossible:

  * the lateral negation is applied twice and cancels, so `evaluate_single_coll`
    receives the raw `planning_traj`;
  * a horizon at which the ground-truth ego box itself collides is excluded from
    scoring, which is why the baseline's collision total is 82 rather than higher.

Usage (from the root of the UniAD working tree):

    python3 analysis/verify_overlap_depth.py --out results/overlap_depth.json
"""
import argparse
import gc
import io
import json

import mmcv
import numpy as np
import torch
from mmcv import Config
from scipy.stats import mannwhitneyu
from skimage.draw import polygon

import projects.mmdet3d_plugin  # noqa: F401
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.uniad.dense_heads.planning_head_plugin import PlanningMetric

torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")

CONFIG = "projects/configs/stage2_e2e/base_e2e.py"


def load_traj(path):
    b = mmcv.load(path)["bbox_results"]
    tr = np.stack([f["planning_traj"].detach().cpu().numpy()[0]
                   for f in b]).astype(np.float64)
    del b
    gc.collect()
    return tr


def footprint_cells(metric, traj):
    """The 32 rasterised cells per horizon, exactly as evaluate_single_coll builds them."""
    pts = np.array([
        [-metric.H / 2. + 0.5, metric.W / 2.],
        [metric.H / 2. + 0.5, metric.W / 2.],
        [metric.H / 2. + 0.5, -metric.W / 2.],
        [-metric.H / 2. + 0.5, -metric.W / 2.],
    ])
    pts = (pts - metric.bx.cpu().numpy()) / (metric.dx.cpu().numpy())
    pts[:, [0, 1]] = pts[:, [1, 0]]
    rr, cc = polygon(pts[:, 1], pts[:, 0])
    rc = np.concatenate([rr[:, None], cc[:, None]], axis=-1)

    t = torch.tensor(traj, dtype=torch.float64).clone()      # clone: the metric
    n_future, _ = t.shape                                    # mutates its input
    trajs = t.view(n_future, 1, 2)
    trajs[:, :, [0, 1]] = trajs[:, :, [1, 0]]
    trajs = trajs / metric.dx
    trajs = trajs.cpu().numpy() + rc                          # (n_future, 32, 2)
    r = np.clip(trajs[:, :, 0].astype(np.int32), 0, metric.bev_dimension[0] - 1)
    c = np.clip(trajs[:, :, 1].astype(np.int32), 0, metric.bev_dimension[1] - 1)
    return r, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="work_dirs/R-0072_2026-08-17_baseline_full")
    ap.add_argument("--arm", default="work_dirs/R-0073_2026-08-17_quantfix_full")
    ap.add_argument("--out", default="results/overlap_depth.json")
    a = ap.parse_args()

    B = load_traj(a.base + "/results.pkl")
    liveB = {r["idx"]: np.array(r["obj_box_col"]) > 0
             for r in json.load(open(a.base + "/per_frame.json"))}
    liveF = {r["idx"]: np.array(r["obj_box_col"]) > 0
             for r in json.load(open(a.arm + "/per_frame.json"))}

    frames = sorted(i for i, v in liveB.items() if v.any())
    print("%d frames carry at least one scored baseline collision" % len(frames))

    ds = build_dataset(Config.fromfile(CONFIG).data.test)
    metric = PlanningMetric()

    removed, survived = [], []
    for idx in frames:
        d = ds.prepare_test_data(idx)
        seg = d["gt_segmentation"]
        seg = (seg[0] if isinstance(seg, list) else seg)[[1, 2, 3, 4, 5, 6]]
        r, c = footprint_cells(metric, B[idx])
        for t in range(6):
            if not liveB[idx][t]:
                continue
            n_cells = int(seg[t].numpy()[r[t], c[t]].sum())
            (removed if not liveF[idx][t] else survived).append(n_cells)

    removed, survived = np.array(removed), np.array(survived)
    u, p = mannwhitneyu(removed, survived, alternative="two-sided")
    out = {
        "n_frames_examined": len(frames),
        "n_collisions_total": int(len(removed) + len(survived)),
        "n_removed": int(len(removed)),
        "n_survived": int(len(survived)),
        "footprint_cells_total": 32,
        "median_overlap_removed": float(np.median(removed)),
        "median_overlap_survived": float(np.median(survived)),
        "mean_overlap_removed": float(removed.mean()),
        "mean_overlap_survived": float(survived.mean()),
        "mannwhitney_u": float(u),
        "mannwhitney_p": float(p),
        "overlap_removed": removed.tolist(),
        "overlap_survived": survived.tolist(),
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("collisions %d = removed %d + survived %d"
          % (out["n_collisions_total"], out["n_removed"], out["n_survived"]))
    print("median overlap, of 32 cells: removed %.1f, survived %.1f"
          % (out["median_overlap_removed"], out["median_overlap_survived"]))
    print("Mann-Whitney U = %.1f, p = %.3g" % (u, p))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
