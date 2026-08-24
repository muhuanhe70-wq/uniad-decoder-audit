# R-0083: ablation of the coordinate-truncation fix.
#
# Keeps the whole-metre quantisation but removes its systematic inward pull, by rounding to
# the nearest integer instead of truncating towards zero. Against R-0072 (truncate, mean
# error 0.5 m towards the ego) and R-0073 (exact, 0 m), this arm sits in between at 0.25 m
# with no preferred direction, and so answers the question neither of those two can:
# is the L2 gain of R-0073 explained by removing the inward pull, or does it need sub-metre
# resolution? Single variable against R-0073.
_base_ = ['./base_e2e.py']
model = dict(planning_head=dict(
    enable_ood_injection=False,
    fix_occ_quantization=True,
    occ_round_to_int=True,
))
