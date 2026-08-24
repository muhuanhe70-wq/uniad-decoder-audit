# R-0084: the planner's raw output, with the collision optimiser switched off entirely.
#
# Not an arm to be compared against the baseline on planning metrics -- without the optimiser
# the trajectory is simply the network's own output. Its purpose is to be the REFERENCE the
# other arms are measured against: |arm - raw| is exactly the displacement the collision
# optimiser applies, so comparing that magnitude between the published arm and the corrected
# one answers, with a number rather than an argument, whether correcting the coordinates makes
# the planner avoid LESS. Li et al. (2024) make that question unavoidable for any open-loop L2
# gain, and every other way we have of addressing it is indirect.
_base_ = ['./base_e2e.py']
model = dict(planning_head=dict(
    enable_ood_injection=False,
    use_col_optim=False,
))
