"""Single definition of the OOD detector's spatial gate.

Before 2026-08-14 this gate was copy-pasted into nine analysis scripts and into
`planning_head.compute_ood_cost_points`, every copy written as

    (xy[:, 0] > 0) & (xy[:, 0] < 30) & (xy[:, 1] > -8) & (xy[:, 1] < 8)

on the assumption that column 0 is the forward axis. It is not.

AXIS CONVENTION, established empirically and not by convention-guessing:
`boxes_2d_xy` (and `track_bboxes_xy`, both from `boxes_3d.gravity_center[:, :2]`)
has **column 0 = lateral, column 1 = forward**. `sdc_traj_all` / `planning_traj`
use the same convention, which is why the Gaussian cost can subtract them directly.

Evidence:
  * 6,010 track_ids matched across consecutive frames while the ego advanced 3.80m
    per step: column 1 changed -3.67m (objects stream backwards past a moving ego),
    column 0 changed +0.03m.
  * The ego's ground-truth driven path never comes within 1m of a detected object
    under this reading; under the transposed reading 13.3% of moving frames would
    have the human driving through another vehicle.
  * Gated objects, resolved into the ego's true driving frame, form a forward
    corridor only under this reading.

The old copies therefore implemented a corridor 0-30m to ONE SIDE with only +/-8m
fore/aft -- not the forward corridor Sec. 3.2, Algorithm 1 and Fig. 1 describe, and
not the one `generate_trainval_risk.py` mines the risk labels with. Correcting it is
worth +11.48pp AUROC on the real inference path, 95% CI [+8.74, +14.17]
(`results/spatial_gate_axis_check.json`).

Import `spatial_gate` from here rather than re-deriving it. If the ranges ever
change, they change in one place.
"""
import numpy as np

# Deployed corridor, in metres. These are physical ranges, NOT column indices.
FORWARD_RANGE = (0.0, 30.0)     # ahead of the ego
LATERAL_RANGE = (-8.0, 8.0)     # either side of the ego

LAT_COL = 0    # boxes_2d_xy column holding the lateral coordinate
FWD_COL = 1    # boxes_2d_xy column holding the forward coordinate


def spatial_gate(xy, forward_range=FORWARD_RANGE, lateral_range=LATERAL_RANGE):
    """Boolean mask over rows of `xy` (N,2) selecting objects inside the corridor.

    Accepts numpy arrays or torch tensors; returns the same kind of boolean mask.
    """
    fwd = xy[:, FWD_COL]
    lat = xy[:, LAT_COL]
    return ((fwd > forward_range[0]) & (fwd < forward_range[1])
            & (lat > lateral_range[0]) & (lat < lateral_range[1]))


def legacy_transposed_gate(xy):
    """The pre-2026-08-14 gate, kept only so the defect can be reproduced on demand.

    Do not use this for any new result. It reads column 0 as forward and column 1 as
    lateral, which is why it selected a sideways corridor.
    """
    return ((xy[:, 0] > FORWARD_RANGE[0]) & (xy[:, 0] < FORWARD_RANGE[1])
            & (xy[:, 1] > LATERAL_RANGE[0]) & (xy[:, 1] < LATERAL_RANGE[1]))


def _self_test():
    """An object 10m directly ahead must pass; the same object 10m to the side must not."""
    ahead = np.array([[0.0, 10.0]])       # lateral 0, forward 10
    beside = np.array([[10.0, 0.0]])      # lateral 10, forward 0
    behind = np.array([[0.0, -10.0]])     # lateral 0, forward -10
    assert spatial_gate(ahead)[0], "object directly ahead must pass the gate"
    assert not spatial_gate(beside)[0], "object 10m to the side must not pass"
    assert not spatial_gate(behind)[0], "object behind must not pass"
    # and the legacy gate must get exactly these backwards, which is the whole point
    assert not legacy_transposed_gate(ahead)[0]
    assert legacy_transposed_gate(beside)[0]
    print("ood_spatial_gate self-test passed")


if __name__ == "__main__":
    _self_test()
