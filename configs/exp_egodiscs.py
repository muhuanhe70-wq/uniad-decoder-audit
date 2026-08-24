# UniAD + multi-disc ego footprint in the collision cost. No OOD layer.
#
# The published cost evaluates collision avoidance at the trajectory POINT while the benchmark
# scores a 4.084 x 1.85 m BOX offset +0.5 m forward. With sigma = 1 m an obstacle touching the
# front bumper (2.54 m from the point) contributes 3.95 % of the cost's peak -- less than the
# reference-tracking term charges for a 0.28 m detour. Discs at (-1.0, 0.5, 2.0) m put that
# same contact 0.54 m from the leading disc, i.e. 87 % of peak.
#
# The per-disc contribution is divided by the disc count, so the total cost weight is
# unchanged: this arm varies footprint GEOMETRY only, not cost magnitude.
_base_ = ['./base_e2e.py']
model = dict(planning_head=dict(
    enable_ood_injection=False,
    ego_discs=[-1.0, 0.5, 2.0],
))
