"""The strongest non-learned baseline: the labelling rule itself, computed from perception.

WHY THIS IS THE TEST THAT MATTERS
---------------------------------
The hazard labels are a hand-written rule applied to ground truth. A reviewer's first question
is therefore not "why not TTC" but the sharper form: "your rule is a closed-form function of
position, velocity, class and visibility --- the tracker estimates three of those four, so why
does a learned head beat simply running your own rule on the tracker's output?"

This script runs exactly that. Every threshold is copied verbatim from
`generate_object_level_risk.py`:

  kinematic   class in vehicle.* / human.pedestrian, dist <= 30 m, forward > 0,
              |lateral| < 8 m, closing > 0.5 m/s, TTC = dist/closing < 2.5 s
  occlusion   class in {traffic_cone, barrier, construction_vehicle}, dist <= 25 m

with two substitutions forced by what perception can actually observe:

  * ego velocity comes from consecutive ego poses rather than the CAN bus (this is what the
    corrected inverse-TTC baseline already uses; omitting it was an error that made an earlier
    version of this comparison unfair, and it cost 19 pp of the baseline's object AUROC);
  * the occlusion rule's VISIBILITY gate has no perception counterpart. The tracker emits no
    visibility estimate, so the replica keeps the class and distance gates and drops the
    visibility one. This is not a handicap imposed by us --- it is the structural reason a
    rule-based system cannot reproduce this half of the label, and the comparison exists to
    quantify exactly that.

Reported as a binary decision (precision/recall/F1, which is what a rule produces) and as a
graded score (AUROC/top-1, so it is comparable with the learned scorers).
"""
import json
import os

import mmcv
import numpy as np
import torch

import ood_split_v2 as sm
import train_temporal_head as T
from ood_spatial_gate import spatial_gate

CACHE = "cache_ext_baselines.npz"
LABELS = "cache_labels3d.npz"

# verbatim from generate_object_level_risk.py
MAX_DISTANCE_KINEMATIC, LATERAL_GATE = 30.0, 8.0
TTC_THRESHOLD, MIN_CLOSING_SPEED = 2.5, 0.5
MAX_DISTANCE_OCC = 25.0
KIN_CLASSES = {0, 1, 2, 3, 4, 6, 7, 8}        # vehicle.* and human.pedestrian
OCC_CLASSES = {9, 5, 2}                        # traffic_cone, barrier, construction_vehicle


def build_labels(TOK):
    out = []
    for t in TOK:
        d = torch.load(sm.feature_path(t), map_location="cpu")
        xy = d["boxes_2d_xy"].to(torch.float32).numpy().astype(np.float64)
        out.append(d["labels_3d"].numpy()[spatial_gate(xy)])
    arr = np.concatenate(out).astype(np.int16)
    np.savez_compressed(LABELS, L=arr)
    return arr


def ego_speed(TOK):
    inf = {}
    for f in ("data/infos/nuscenes_infos_temporal_train.pkl",
              "data/infos/nuscenes_infos_temporal_val.pkl"):
        for i in mmcv.load(f)["infos"]:
            inf[i["token"]] = i
    v = np.zeros(len(TOK))
    for k, t in enumerate(TOK):
        i = inf.get(t)
        if i is None:
            continue
        j = inf.get(i.get("next", "") or i.get("prev", ""))
        if j is None:
            continue
        dt = abs(j["timestamp"] - i["timestamp"]) * 1e-6
        if dt > 1e-3:
            v[k] = np.linalg.norm(np.array(j["ego2global_translation"][:2])
                                  - np.array(i["ego2global_translation"][:2])) / dt
    return v


def prf(pred, y):
    tp = float((pred & (y > 0)).sum()); fp = float((pred & (y == 0)).sum())
    fn = float((~pred & (y > 0)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return 100 * p, 100 * r, 100 * 2 * p * r / max(p + r, 1e-9)


def main():
    d = dict(np.load(CACHE, allow_pickle=False))
    V, XY, O, RULE, sz = d["V"], d["XY"], d["O"], d["RULE"], d["sz"]
    Y = d["Y"]; TOK = [str(t) for t in d["TOK"]]
    L = np.load(LABELS)["L"] if os.path.exists(LABELS) else build_labels(TOK)
    assert len(L) == len(O), f"class array {len(L)} != objects {len(O)}"
    off = np.concatenate([[0], np.cumsum(sz)])

    es = ego_speed(TOK)
    eso = np.concatenate([np.full(sz[i], es[i]) for i in range(len(sz))])
    lat, fwd = XY[:, 0], XY[:, 1]
    dist = np.maximum(np.linalg.norm(XY, axis=1), 1e-3)
    rel = V.copy(); rel[:, 1] -= eso
    closing = -(rel * XY).sum(1) / dist
    is_kin = np.isin(L, list(KIN_CLASSES))
    is_occ = np.isin(L, list(OCC_CLASSES))

    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(closing > MIN_CLOSING_SPEED, dist / np.maximum(closing, 1e-6), np.inf)
    kin_hit = (is_kin & (dist <= MAX_DISTANCE_KINEMATIC) & (fwd > 0)
               & (np.abs(lat) < LATERAL_GATE) & (closing > MIN_CLOSING_SPEED)
               & (ttc < TTC_THRESHOLD))
    occ_hit = is_occ & (dist <= MAX_DISTANCE_OCC)
    both = kin_hit | occ_hit

    print(f"objects {len(O)}   hazardous {int(O.sum())} "
          f"(kinematic {int((RULE==1).sum())}, occlusion {int((RULE==2).sum())})")
    print(f"rule replica fires on: kinematic {int(kin_hit.sum())}, "
          f"occlusion {int(occ_hit.sum())}, either {int(both.sum())}\n")

    print("BINARY rule replica vs the GT-derived labels")
    print(f"{'':26s}{'precision':>11}{'recall':>9}{'F1':>8}")
    for lab, pred, mask in (("kinematic rule -> kin haz", kin_hit, (O == 0) | (RULE == 1)),
                            ("occlusion rule -> occ haz", occ_hit, (O == 0) | (RULE == 2)),
                            ("combined -> any hazard  ", both, np.ones(len(O), bool))):
        p, r, f1 = prf(pred[mask], O[mask])
        print(f"   {lab:26s}{p:10.2f}%{r:8.2f}%{f1:7.2f}%")

    # graded version, comparable with the learned scorers
    kin_s = np.where(kin_hit | ((closing > MIN_CLOSING_SPEED) & is_kin & (fwd > 0)),
                     1.0 / np.maximum(ttc, 1e-3), 0.0)
    kin_s = np.clip(kin_s, 0, 10) / 10.0
    occ_s = np.where(is_occ, np.clip((MAX_DISTANCE_OCC - dist) / MAX_DISTANCE_OCC, 0, 1), 0.0)
    grade = np.maximum(kin_s, occ_s)

    def top1(s):
        h = []
        for i in range(len(sz)):
            o = O[off[i]:off[i + 1]]
            if len(o) < 2 or o.sum() == 0:
                continue
            h.append(o[int(np.argmax(s[off[i]:off[i + 1]]))])
        return 100 * np.mean(h)

    def fp(s):
        return np.array([s[off[i]:off[i + 1]].max() for i in range(len(sz))])

    kinm = (O == 0) | (RULE == 1); occm = (O == 0) | (RULE == 2)
    print(f"\nGRADED scorers, comparable with Table of external baselines")
    print(f"{'scorer':34s}{'frame':>9}{'object':>9}{'top-1':>8}{'kin':>8}{'occ':>8}")
    rows = {}
    for lab, s in (("rule replica (graded)", grade),
                   ("  ...kinematic arm only", kin_s),
                   ("  ...occlusion arm only", occ_s)):
        row = (100 * T.auroc(fp(s), Y), 100 * T.auroc(s, O), top1(s),
               100 * T.auroc(s[kinm], O[kinm]), 100 * T.auroc(s[occm], O[occm]))
        print(f"{lab:34s}{row[0]:8.2f}%{row[1]:8.2f}%{row[2]:7.2f}%{row[3]:7.2f}%{row[4]:7.2f}%")
        rows[lab.strip()] = row
    ours = (88.69, 92.05, 88.72, 96.78, 85.70)
    print(f"{'Object-level MLP (ours)':34s}{ours[0]:8.2f}%{ours[1]:8.2f}%{ours[2]:7.2f}%"
          f"{ours[3]:7.2f}%{ours[4]:7.2f}%")
    g = rows["rule replica (graded)"]
    print(f"\nours minus the rule replica:")
    for i, n in enumerate(("frame AUROC", "object AUROC", "top-1", "kinematic", "occlusion")):
        print(f"   {n:14s} {ours[i]-g[i]:+7.2f} pp")

    json.dump({"binary": {"kin": prf(kin_hit, O), "occ": prf(occ_hit, O),
                          "combined": prf(both, O)},
               "graded": {k: list(v) for k, v in rows.items()}, "ours": list(ours)},
              open("results/rule_replica.json", "w"), indent=2)
    print("\nwrote results/rule_replica.json")


if __name__ == "__main__":
    main()
