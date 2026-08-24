"""Side-by-side trajectories, before and after the coordinate-truncation fix.

Everything is drawn in the frame the benchmark actually scores in, reconstructed from
PlanningMetric's own rasterisation rather than assumed:

    segmentation[t, r, c] occupied  <->  forward = 0.5*r - 49.75 ,  lateral = 0.5*c - 49.75

Note the lateral sign. PlanningMetric.update negates the trajectory's x, and evaluate_coll
then negates it AGAIN, so evaluate_single_coll receives the original lateral sign and the
raster axis is NOT mirrored. Getting this wrong puts the ego box on the wrong side of the
road, which is how the error was caught: boxes the benchmark scores as colliding appeared to
sit in empty space. The mapping agrees with the planning head's own index-to-metre formula
(i - 100)*0.5 + 0.25 -- index 100 lands at 0.25 m in both.

The ego box drawn at each waypoint is the same 4.084 x 1.85 m box, offset +0.5 m forward, that
the benchmark rasterises, so a box that visibly overlaps a shaded cell is a scored collision
and not an illustration.

Usage:  python plot_qualitative.py 2966 3116 2949
"""
import io
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.draw import polygon
import numpy as np
import torch
from mmcv import Config

import projects.mmdet3d_plugin  # noqa: F401
from mmdet3d.datasets import build_dataset
import mirror_control as M

torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")

BX, DX = -49.75, 0.5
EGO_L, EGO_W, EGO_OFF = 4.084, 1.85, 0.5


def occ_to_metres(seg_t):
    """(200,200) occupancy -> arrays of lateral, forward coordinates of occupied cells."""
    r, c = np.nonzero(seg_t)
    return DX * c + BX, DX * r + BX             # lateral, forward


_pts = np.array([[-EGO_L/2 + EGO_OFF,  EGO_W/2], [EGO_L/2 + EGO_OFF,  EGO_W/2],
                 [EGO_L/2 + EGO_OFF, -EGO_W/2], [-EGO_L/2 + EGO_OFF, -EGO_W/2]])
_pts = (_pts - BX) / DX
_pts[:, [0, 1]] = _pts[:, [1, 0]]
_rr, _cc = polygon(_pts[:, 1], _pts[:, 0])
RC = np.stack([_rr, _cc], 1)


def footprint_cells(lat, fwd):
    """The cells the benchmark actually rasterises the ego box into at one waypoint.

    This is not the geometric rectangle: evaluate_single_coll adds the fractional waypoint
    offset to the integer polygon and then casts with astype(int32), so the scored footprint
    is the set of grid cells below, which can reach up to half a cell beyond the rectangle.
    Drawing the rectangle instead of these cells makes scored collisions appear to sit in
    empty space -- which is how this was noticed."""
    r = (fwd / DX + RC[:, 0]).astype(np.int32)
    c = (lat / DX + RC[:, 1]).astype(np.int32)
    return r, c


def cells_to_metres(r, c):
    return DX * c + BX, DX * r + BX              # lateral, forward


def main():
    idxs = [int(a) for a in sys.argv[1:]] or [2966]
    T0 = M.load_traj("work_dirs/R-0072_2026-08-17_baseline_full/results.pkl")
    T1 = M.load_traj("work_dirs/R-0073_2026-08-17_quantfix_full/results.pkl")
    ds = build_dataset(Config.fromfile("projects/configs/stage2_e2e/base_e2e.py").data.test)

    p0 = __import__("json").load(open("work_dirs/R-0072_2026-08-17_baseline_full/per_frame.json"))
    p1 = __import__("json").load(open("work_dirs/R-0073_2026-08-17_quantfix_full/per_frame.json"))
    p0 = {r["idx"]: np.asarray(r["obj_box_col"]).reshape(-1) > 0 for r in p0}
    p1 = {r["idx"]: np.asarray(r["obj_box_col"]).reshape(-1) > 0 for r in p1}
    HOR = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    fig, axes = plt.subplots(1, len(idxs), figsize=(3.9 * len(idxs), 4.3), squeeze=False)
    for ax, idx in zip(axes[0], idxs):
        d = ds.prepare_test_data(int(idx))
        seg = d["gt_segmentation"]
        seg = (seg[0] if isinstance(seg, list) else seg)[[1, 2, 3, 4, 5, 6]].numpy()
        gt = np.array(d["sdc_planning"]).reshape(-1, 6, 3)[0][:, :2]

        # the horizon the published arm collides at and the corrected one does not
        t = int(np.argmax(p0[idx] & ~p1[idx]))

        lat, fwd = occ_to_metres(seg[t])
        ax.scatter(lat, fwd, s=9, marker="s", color="#8a8f98", linewidths=0, zorder=1,
                   label=f"occupied at {HOR[t]:.1f}s")

        for traj, col, lab in ((T0[idx], "#c0392b", "as published"),
                               (T1[idx], "#1f6fb4", "coordinates corrected")):
            ax.plot([0] + list(traj[:, 0]), [0] + list(traj[:, 1]), "-", color=col,
                    lw=1.5, label=lab, zorder=4)
            r, c = footprint_cells(traj[t, 0], traj[t, 1])
            fl, ff = cells_to_metres(r, c)
            hit = seg[t, np.clip(r, 0, 199), np.clip(c, 0, 199)] > 0
            ax.scatter(fl, ff, s=26, marker="s", facecolors="none", edgecolors=col,
                       linewidths=0.7, zorder=4)
            if hit.any():
                ax.scatter(fl[hit], ff[hit], s=46, marker="s", color=col, alpha=0.85,
                           linewidths=0, zorder=6, label="scored collision")
            ax.plot(traj[t, 0], traj[t, 1], "o", color=col, ms=4, zorder=5)

        ax.plot([0] + list(gt[:, 0]), [0] + list(gt[:, 1]), "--", color="k", lw=1.2,
                label="human trajectory", zorder=3)
        ax.plot(0, 0, marker="*", ms=12, color="k", zorder=6)

        cx, cy = T0[idx][t]
        ax.set_xlim(cx - 7.5, cx + 7.5); ax.set_ylim(min(-3, cy - 9), cy + 9)
        ax.set_title(f"frame {idx}: collision at {HOR[t]:.1f}s removed", fontsize=8.5)
        ax.set_xlabel("lateral (m)", fontsize=8); ax.set_ylabel("forward (m)", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.set_aspect("equal"); ax.grid(alpha=0.25, lw=0.4)
        ax.legend(fontsize=6.8, loc="upper left", framealpha=0.92)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"paper_drafts/phase8-paper-20260821-decoder-audit/manuscript/"
                    f"figures/qualitative.{ext}", dpi=180, bbox_inches="tight")
    print("wrote figures/qualitative.pdf and .png")


if __name__ == "__main__":
    main()
