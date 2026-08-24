"""Re-derive the open-loop rear-sweep artefact statistic quoted in the 2026-08-21 draft.

CLAIM UNDER TEST. Of the frames where the baseline planner is scored as colliding (box,
3.0 s), in how many was the counterpart agent already BEHIND the ego at t=0? Such an event
is an artefact of open-loop replay: agents occupy the positions they were recorded in, which
presuppose that the ego did what the human driver did. If the counterpart was behind the ego
and the ego moves forward, the ego did not drive into it -- the recorded follower drove
through the ego's box because the ego did not proceed as the recording assumed. No amount of
better planning removes such an event.

The statistic was measured in an earlier session and never written to results/, so it is
recomputed here before being asserted in a submission draft.

DEFINITIONS, stated because they are approximations.
* "Counterpart" = the agent whose t=0 centre is closest to the ego's planned path over the
  six horizon waypoints. Agent futures are not used: nuScenes' converter stores fut_traj
  un-rotated in each agent's own heading frame, so decoding them adds a failure mode this
  statistic does not need. Using t=0 positions is exactly the frozen-world assumption whose
  consequences are the subject of the claim.
* Frames are addressed positionally: per_frame.json's `idx` and `data_infos[idx]` share the
  index, the convention the rest of this project uses (see mirror_control.py).

COORDINATE CONVENTION. gt_boxes column 1 is the ego's forward axis; verified empirically in
this project by checking that moving agents' velocity concentrates there (5.92 m/s on column
1 against 1.86 m/s on column 0 over 109 sampled frames). The planned trajectory's lateral
SIGN relative to gt_boxes is not independently established -- PlanningMetric negates its
column 0 before rasterising -- so the counterpart selection is run under both lateral sign
conventions and the result is only believed if the two agree.
"""
import argparse
import gc
import glob
import io
import json

import numpy as np
import torch
import mmcv
from mmcv import Config

import projects.mmdet3d_plugin  # noqa: F401  -- registers NuScenesE2EDataset
from mmdet3d.datasets import build_dataset

torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")

CONFIG = "projects/configs/stage2_e2e/base_e2e.py"
HOR = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def counterpart_offsets(ds, coll, traj, lateral_sign):
    """Longitudinal offset at t=0 of each collision frame's counterpart agent."""
    offs = []
    for i in coll:
        boxes = np.asarray(ds.data_infos[int(i)]["gt_boxes"], float)
        if boxes.ndim != 2 or len(boxes) == 0:
            offs.append(None)
            continue
        agents = boxes[:, :2].copy()            # (A, 2), col0 lateral, col1 forward
        path = traj[int(i)].copy()              # (6, 2)
        path[:, 0] *= lateral_sign
        # distance from every horizon waypoint to every agent centre
        d = np.linalg.norm(path[:, None, :] - agents[None, :, :], axis=2)  # (6, A)
        j = int(np.argmin(d.min(axis=0)))
        offs.append(float(agents[j, 1]))
    return offs


def summarise(offs):
    v = [o for o in offs if o is not None]
    rear = sum(1 for o in v if o < 0)
    return {
        "n_analysed": len(v),
        "n_skipped_no_agents": len(offs) - len(v),
        "behind_at_t0": rear,
        "ahead_at_t0": len(v) - rear,
        "rear_fraction": (rear / len(v)) if v else None,
        "median_longitudinal_offset_m": float(np.median(v)) if v else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="work_dirs/R-0072_*_baseline_full")
    ap.add_argument("--out", default="results/artefacts.json")
    a = ap.parse_args()

    d0 = sorted(glob.glob(a.base))[0]
    per = {r["idx"]: r for r in json.load(open(f"{d0}/per_frame.json"))}
    hi = len(HOR) - 1
    coll = [i for i in sorted(per) if np.asarray(per[i]["obj_box_col"]).reshape(-1)[hi] > 0]
    print(f"baseline: {d0.split('/')[-1]}", flush=True)
    print(f"box collisions @3.0 s: {len(coll)}", flush=True)

    b = mmcv.load(f"{d0}/results.pkl")["bbox_results"]
    traj = np.stack([f["planning_traj"].detach().cpu().numpy()[0] for f in b]).astype(np.float64)
    del b
    gc.collect()

    print("building dataset ...", flush=True)
    ds = build_dataset(Config.fromfile(CONFIG).data.test)

    res = {"baseline": d0.split("/")[-1], "n_collision_frames_3s": len(coll)}
    for sign, name in ((+1.0, "lateral_as_is"), (-1.0, "lateral_negated")):
        s = summarise(counterpart_offsets(ds, coll, traj, sign))
        res[name] = s
        print(f"{name:16s} behind {s['behind_at_t0']:3d} / {s['n_analysed']:3d} "
              f"= {100*s['rear_fraction']:.1f}%   median offset "
              f"{s['median_longitudinal_offset_m']:+.2f} m", flush=True)

    agree = (res["lateral_as_is"]["behind_at_t0"] == res["lateral_negated"]["behind_at_t0"])
    res["conventions_agree"] = bool(agree)
    print(f"\nconventions agree: {agree}", flush=True)
    if not agree:
        print("DO NOT QUOTE: the two lateral sign conventions disagree; the counterpart "
              "selection is not robust and the statistic needs a definition that does not "
              "depend on it.", flush=True)

    json.dump(res, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
