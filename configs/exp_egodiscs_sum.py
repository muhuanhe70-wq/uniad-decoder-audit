# UniAD + multi-disc ego footprint, STANDARD multi-circle weighting. No OOD layer.
#
# Same geometry as exp_egodiscs.py, but each disc carries the full cost weight instead of
# 1/3 of it. R-0080 used the divided form so that footprint geometry could be isolated from
# repulsion strength; the price is that a bumper contact excites essentially one disc and
# therefore produces a THIRD of the force the point model applies to a centre contact. That
# makes R-0080 a lower bound on the mechanism rather than a fair test of it.
#
# Summing is what the multi-circle approximation normally does, and it is NOT the same as
# raising alpha_collision: the field becomes an elongated sausage aligned with the body, so
# it changes where the cost is high, not only how high. The extra repulsion it brings is a
# real confound, and mirror_control.py is the control that separates the two -- it rebuilds a
# per-frame, per-waypoint magnitude-matched counterfactual from this arm's own displacements.
#
# Compare against R-0072 (published point model, no OOD), the same reference R-0080 used.
_base_ = ['./base_e2e.py']
model = dict(planning_head=dict(
    enable_ood_injection=False,
    ego_discs=[-1.0, 0.5, 2.0],
    ego_disc_reduce='sum',
))
