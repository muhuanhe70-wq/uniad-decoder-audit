# Auditing the Planning Decoder of a Reference End-to-End Driving Stack

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

All three hold the per-waypoint displacement magnitude fixed and vary one thing:

```bash
# direction, by reflecting the cross-path component
python controls/mirror_control.py --base work_dirs/baseline --arm work_dirs/quantfix

# direction, sampled uniformly on the circle (K draws per frame)
python controls/rotation_control.py --base work_dirs/baseline --arm work_dirs/quantfix \
    --out results/rotation_control_quantfix.json --k 8

# allocation: which frames the withdrawn avoidance falls on, total held fixed
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

---

## What the numbers do and do not show

**They show** that the correction's displacement is directional and correctly allocated:
against equal-magnitude controls the real correction nets +11 (uniform random direction:
−69.4, 0/8 draws better) and +12 (permuted allocation: −23.0, 0/8 better), both at
p < 10⁻⁵.

**They do not show** a demonstrated safety improvement. Every significant result compares
the arm against *magnitude-matched controls*, not against the baseline. The direct
comparison against the baseline is the collision count, which improves (20 repaired
against 9 introduced) but does **not** reach significance (p = 0.127). We claim only that
collision rate does not worsen.

**The corrected planner does avoid less** — the applied avoidance falls 47%, only 1.0% of
frames avoid more, and 94% of the trajectory change is magnitude rather than direction
(`results/avoidance.json`). That is the fix's mechanism, not an alternative explanation
for it: the obstacles were reported half a metre nearer than they are. The allocation
control is what shows the avoidance was withdrawn on the right frames.

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

## Citation

*(fill in on acceptance)*

## Licence

Scripts in this repository are released under the MIT licence. UniAD and nuScenes remain
under their own licences and are not redistributed here.
