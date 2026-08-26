# What the Decoder Computes and What the Benchmark Scores: Two Implementation Defects in a Reference End-to-End Driving Stack

Code and results for the paper. Everything here runs against the **public**
`uniad_base_e2e` checkpoint on the **full nuScenes validation split** (6,019 frames).
Nothing is retrained.

This repository deliberately does **not** vendor UniAD or nuScenes. It contains the
one-line patch, the controls, the analysis scripts, and the result files behind every
number in the paper, so that each claim can be re-derived on top of an unmodified
upstream checkout.

---

## The defect, in one paragraph

`torch.nonzero` returns `int64`. In `planning_head.py` the two lines that convert
occupancy grid indices to metres write their float results back **into that same int64
tensor**, so every coordinate reaching the collision optimiser is truncated toward zero.
On the 0.5 m BEV grid this displaces every occupied cell by a mean of 0.5 m — one full
cell — and always toward the ego vehicle, because truncation toward zero is toward the
grid origin. Cell `(103, 97)` reaches the solver at `(-1.00, 1.00)` m instead of
`(-1.25, 1.75)` m.

The `+ 0.25` in those lines is a cell-centre offset. Truncation destroys it.

## The fix

```diff
  pos_xy = torch.nonzero(occ_mask[0][cur_t], as_tuple=False)
  pos_xy = pos_xy[:, [1, 0]]
+ pos_xy = pos_xy.to(torch.float64)
  pos_xy[:, 0] = (pos_xy[:, 0] - self.bev_h//2) * 0.5 + 0.25
  pos_xy[:, 1] = (pos_xy[:, 1] - self.bev_w//2) * 0.5 + 0.25
```

**The cast must come before the arithmetic.** Casting afterwards truncates first and then
converts the already-wrong integer, which is a no-op on the defect.

See `patch/0001-fix-occupancy-coordinate-truncation.patch`.

## Result

| | 1 s | 2 s | 3 s | avg |
|---|---|---|---|---|
| UniAD, published | 0.48 | 0.96 | 1.65 | 1.03 |
| our replication | 0.5071 | 0.9812 | 1.6469 | 1.0451 |
| **with the one-line fix** | **0.4418** | **0.9332** | **1.6105** | **0.9952** |

L2 improves at all six horizons; box collision rate falls from 0.294% to 0.260%
(not significant — see below). Averaging convention is UniAD's own (1/2/3 s).

**Rounding instead of casting recovers essentially all of it.** An ablation that rounds to
the nearest cell — keeping whole-metre quantisation but removing its systematic inward
direction — reproduces the fix to within 0.8 mm at every horizon
(`results/quantround.json`). If the exact cast is inconvenient in your fork, a single
`round()` will do.

---

## Reproducing

### 1. Apply the patch to an upstream UniAD checkout

```bash
git clone https://github.com/OpenDriveLab/UniAD && cd UniAD
git apply /path/to/patch/0001-fix-occupancy-coordinate-truncation.patch
```

Follow UniAD's own installation and data-preparation instructions, then place
`uniad_base_e2e.pth` in `ckpts/`.

### 2. Run the arms

Copy `configs/*.py` into `projects/configs/stage2_e2e/`, then for each arm:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/test.py \
    projects/configs/stage2_e2e/exp_baseline.py ckpts/uniad_base_e2e.pth \
    --launcher pytorch --eval bbox --out work_dirs/baseline/results.pkl \
    --show-dir work_dirs/baseline/
python analysis/build_metrics_json.py work_dirs/baseline baseline exp_baseline.py
```

| config | arm |
|---|---|
| `exp_baseline.py` | unmodified baseline |
| *(patched baseline)* | the one-line fix |
| `exp_quantround.py` | rounding ablation |
| `exp_nocoloptim.py` | collision optimiser disabled — gives the raw network trajectory |
| `exp_egodiscs.py`, `exp_egodiscs_sum.py` | three-disc ego footprint, two repulsion gains |

One arm is roughly 7 h on a single RTX 4060. **Run them one at a time** — each holds the
whole nuScenes dataset in-process (`workers_per_gpu=0`), so two concurrent arms need
~26 GB of host RAM and are *slower* in aggregate than running them in sequence
(measured: 2 × 0.0876 vs 0.245 frames/s).

### 3. Controls

The two direction controls hold the per-waypoint displacement magnitude fixed exactly
and vary only direction. The third does not hold magnitude fixed and is withdrawn:

```bash
# direction, by reflecting the cross-path component
python controls/mirror_control.py --base work_dirs/baseline --arm work_dirs/quantfix

# direction, sampled uniformly on the circle (K draws per frame)
python controls/rotation_control.py --base work_dirs/baseline --arm work_dirs/quantfix \
    --out results/rotation_control_quantfix.json --k 8

# allocation: which frames the withdrawn avoidance falls on
# (WITHDRAWN -- see the note below; this control does not hold the total fixed)
python controls/allocation_control.py --base work_dirs/baseline --arm work_dirs/quantfix \
    --raw work_dirs/nocoloptim --out results/allocation_control.json --k 8
```

CPU only, roughly 1 h each when run alone. All three re-score with the benchmark's own
`PlanningMetric.evaluate_coll` rather than a reimplementation, and self-validate the real
arm's recomputed flags against the live per-frame outputs before any control number is
believed (0 mismatches in 6,676–6,678 checks).

### 4. Analysis

```bash
python analysis/analyse_quantfix.py  work_dirs/baseline work_dirs/quantfix
python analysis/analyse_avoidance.py --raw work_dirs/nocoloptim \
    --base work_dirs/baseline --arm work_dirs/quantfix --out results/avoidance.json
python analysis/audit_propagation.py     # needs a GitHub token in ~/.gh_token
```

### Verifying the numbers that are not otherwise in a result file

Several figures the paper quotes were originally read off console output rather than
written to a file. These scripts re-derive them so that every quoted number rests on
something re-runnable:

```bash
python3 analysis/verify_control_invariants.py   # magnitude match + mirror direction cosines
python3 analysis/verify_overlap_depth.py        # how deep the repaired collisions were
python3 analysis/verify_rounding_divergence.py  # exact-cast vs rounding arm, frame by frame
python3 analysis/verify_swerve_geometry.py      # swerves less, or travels less?
python3 analysis/verify_dose_response.py        # cost linear in dose, benefit zero
python3 analysis/verify_bumper_response.py      # the three-disc geometry probe (no data needed)
python3 analysis/audit_allocation_magnitude.py  # the withdrawn control, see above
```

writing `results/control_invariants.json`, `results/overlap_depth.json`,
`results/rounding_divergence.json`, `results/swerve_geometry.json`,
`results/dose_response.json`, `results/bumper_response.json` and
`results/allocation_magnitude_audit.json`. Only
`verify_overlap_depth.py` needs the dataset; it touches 53 frames and takes a few minutes.

Two of these changed what the paper says. The direction cosines and the magnitude-match
residuals did not reproduce under any definition we could reconstruct, so they were
recomputed under a stated one (median over (frame, waypoint) pairs; residuals measured
from the reconstructed trajectory). The rounding-divergence figures turned out to have
been per-axis rather than Euclidean, and were restated. The overlap-depth statistics
reproduced exactly, and the swerve and dose-response figures reproduced to a rounding
digit — except the per-intervention magnitude, which did not reproduce under any
definition and was restated. The bumper-response probe reproduces exactly, but only with
a stationary reference; the script explains why a moving one changes the answer.

---

## What the numbers do and do not show

**They show** that the correction's displacement is *aimed*: against equal-magnitude
controls that keep the size of the move and change only its bearing, the real correction
nets +11 against −69.4 for a uniform random direction, with 0/8 draws better, at p < 10⁻⁵.

**They also show** that it is correctly *allocated* across frames: against permuted
allocations carrying an identical total displacement, the real allocation nets +12 against
-15.0, 0/8 draws better, p = 6e-5. An earlier control that permuted retention factors
instead of magnitudes is confounded and withdrawn — see above.

**They do not show** a demonstrated safety improvement. Every significant result compares
the arm against *magnitude-matched controls*, not against the baseline. The direct
comparison against the baseline is the collision count, which improves (20 repaired
against 9 introduced) but does **not** reach significance (p = 0.127). We claim only that
collision rate does not worsen.

**The corrected planner does avoid less** — the applied avoidance falls 47%, only 1.0% of
frames avoid more, and 94% of the trajectory change is magnitude rather than direction
(`results/avoidance.json`). That is the fix's mechanism, not an alternative explanation
for it: the obstacles were reported half a metre nearer than they are. That the avoidance
was withdrawn on the *right* frames is established by the magnitude-matched allocation
control above, not by the earlier retention-permuting one, which is withdrawn.

## Propagation audit

`results/propagation_audit.json` records a GitHub code search over two identifiers unique
to this collision stage, with the source of all 91 matching repositories fetched and
inspected. 78 carry the defect, 0 have fixed it, 13 do not contain this code path (they
reimplement the stage rather than vendoring it). Re-run with
`analysis/audit_propagation.py`.

This is a lower bound: GitHub code search indexes only default branches of public
repositories.

## Layout

```
patch/      the one-line fix
controls/   mirror, uniform-random-direction, allocation
analysis/   per-arm analysis, avoidance measurement, propagation audit
configs/    one config per arm reported in the paper
results/    the result files behind every number in the paper
```

`results/arm_metrics.json` holds the per-horizon L2 and box-collision counts for every
arm in the paper, so the headline table can be rebuilt without re-running anything.

### The allocation control: one version withdrawn, one that works

`controls/allocation_control.py` was intended to separate "withdrew avoidance on the right
frames" from "withdrew less avoidance in total". It does not. It permutes the per-frame
retention factor `s_i`, which preserves the multiset of retention *factors* but not the total
applied displacement `sum_i s_i * ||A^B_i||`, because `s` and `||A^B||` correlate at
`r = +0.468`. The permuted arms apply 67.3% of the reference's avoidance, and every unit of the
reported +35.0 advantage arises on frames where they applied less:

```bash
python3 analysis/audit_allocation_magnitude.py     # no GPU, re-scores nothing
```

writes `results/allocation_magnitude_audit.json`. The paper reports that version as one that
did not work and draws no conclusion from it.

`controls/allocation_control_magnitude.py` is the version that does work. It permutes the
applied magnitudes `m_i = s_i * ||A^B_i||` rather than the factors, so the multiset of
displacements — and hence the total — is reproduced exactly (528.083 m on both sides,
agreeing to 1e-13 m). It permutes only among frames with `||A^B_i|| >= 0.10 m`, because
below that the avoidance *direction* is numerical noise and handing such a frame a large
magnitude would be a random-direction perturbation rather than a reallocation; unrestricted,
20% of (frame, draw) pairs get a scale factor above 10. The threshold keeps 1,681 of 3,338
frames and 97.8% of the applied avoidance.

```bash
python controls/allocation_control_magnitude.py   # ~35 min, CPU only
```

Result: real allocation **+12** cells against **-15.0** for permuted ones (range -21 to -9),
**0/8** draws better, advantage 27.0 over 63 informative frames, p = 6e-5, self-validation
0 mismatches in 3,362 checks. The advantage is +24.6 on frames where a permutation withdrew
avoidance the reference kept and only +2.4 where it added avoidance the reference did not,
so placement — not amount — is what carries it.

The direction controls (mirror, negation, uniform random rotation) were never affected by
any of this: they hold per-waypoint magnitude fixed exactly.

The paper's two data figures are plotted from these files rather than drawn by hand:

```bash
python3 analysis/make_data_figures.py <output_dir>
```

writes `tradeoff.dat` (Figure 5, the avoidance/L2/collision trade-off) and `controls.dat`
plus `controls_axis.tex` (Figure 4, every magnitude-matched control draw). The manuscript
reads those tables directly, so no figure coordinate is retyped.

## Citation

*(fill in on acceptance)*

## Licence

Scripts in this repository are released under the MIT licence. UniAD and nuScenes remain
under their own licences and are not redistributed here.
