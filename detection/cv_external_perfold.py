"""Published OOD scorers and a non-learned kinematic scorer, under the corrected protocol.

WHY
---
The object-level-versus-frame-level comparison is internal: both arms are ours. An internal
ablation can show that a design choice matters; it cannot show the result is worth anything
against what the field already does. This script supplies the external anchor, on the
corrected protocol (all-weather labels, scene-disjoint 5-fold CV), and it reports the two
things that matter separately:

  frame AUROC   is this frame hazardous            (max-pooled over the frame's objects)
  object AUROC  which object is the hazard         (the localization claim)
  top-1         is the highest-scoring object hazardous, on frames with >= 1 hazard and
                >= 2 objects -- the quantity the planner consumes, since the repulsion
                field is placed on the argmax

METHODS
-------
  MSP-analogue      Hendrycks & Gimpel 2017. UniAD's track head is sigmoid focal and emits no
                    class logit vector, so the direct analogue is its max-class detection
                    confidence; score = -confidence. Label-free.
  kNN               Sun et al. 2022, k=50, L2-normalised features, distance to in-distribution
                    (normal-frame) training tracks. Label-free apart from choosing the ID set.
  Mahalanobis       Lee et al. 2018, class-conditional with pooled within-class covariance.
  LDA (LLR)         Supervision-matched linear control: tied-covariance Gaussian log-likelihood
                    ratio on the same labels. Isolates "MLP + VOS" from "having labels".
  inverse TTC       NOT learned and NOT feature-based: from the tracker's own position and
                    velocity, closing = -(v . p)/|p|, score = closing/|p| when closing > 0.
                    This is the baseline a reviewer reaches for first -- if a learned head
                    cannot beat arithmetic on the tracker output, it has no reason to exist.
  Object-level MLP  ours.

ONE ASYMMETRY, STATED
---------------------
MSP, kNN and Mahalanobis are label-free; ours is trained with risk labels. Comparing them
without saying so flatters us. LDA is the supervision-matched control and is the one to read
against ours. Inverse TTC is label-free but encodes the kinematic half of the labelling rule
directly, so it is expected to be strong on kinematic hazards and blind to occlusion hazards
-- which are 69 % of them. Results are therefore also broken out by hazard type.
"""
import json
import os

import numpy as np
import torch

import ood_split_v2 as sm
import train_temporal_head as T
from ood_spatial_gate import spatial_gate
from cv_temporal_head import NFOLD, scene_folds, group, boot_diff

CACHE = "cache_ext_baselines.npz"
FLIP = np.array([-1.0, 1.0])
MATCH_R = 2.0


def build():
    """Extends cache_allweather.npz with the per-object velocity and xy the TTC baseline needs,
    and with a per-object hazard-rule tag so results can be split by hazard type."""
    import mmcv
    lab = json.load(open("risk_objects_v3.json"))
    risk_all = set(lab)
    ntr, nte, rtr, rte = sm.get_split()
    import gc
    b = mmcv.load("work_dirs/R-0006_2026-08-04_baseline_pf_a/results.pkl")["bbox_results"]
    val = set(f["token"] for f in b); del b; gc.collect()
    toks = [t for t in (list(ntr) + list(nte) + list(rtr) + list(rte)) if t not in val]
    print(f"loading {len(toks)} frames ...", flush=True)
    F, V, XY, O, RULE, Y, TOK, SZ = [], [], [], [], [], [], [], []
    for t in toks:
        try:
            d = torch.load(sm.feature_path(t), map_location="cpu")
        except Exception:
            continue
        q = d["track_query"].to(torch.float32)
        if q.shape[0] == 0:
            continue
        xy = d["boxes_2d_xy"].to(torch.float32).numpy().astype(np.float64)
        g = spatial_gate(xy)
        if not g.any():
            continue
        f = torch.cat([q, d["logits"].to(torch.float32)], -1).numpy().astype(np.float32)[g]
        v = d["velocity"].to(torch.float32).numpy().astype(np.float32)[g]
        p = xy[g].astype(np.float32)
        ol = np.zeros(len(f), np.float32); rl = np.zeros(len(f), np.int8)
        rec = lab.get(t)
        if rec:
            pts = np.array([np.array(o["ego_xy_lat_fwd"]) * FLIP for o in rec["objects"]])
            kinds = np.array([1 if o["rule"] == "kinematic" else 2 for o in rec["objects"]])
            if len(pts):
                dd = np.linalg.norm(p[:, None, :] - pts[None, :, :], axis=-1)
                j = dd.argmin(1); hit = dd.min(1) < MATCH_R
                ol = hit.astype(np.float32); rl = np.where(hit, kinds[j], 0).astype(np.int8)
        F.append(f); V.append(v); XY.append(p); O.append(ol); RULE.append(rl)
        Y.append(1.0 if t in risk_all else 0.0); TOK.append(t); SZ.append(len(f))
    out = {"F": np.concatenate(F), "V": np.concatenate(V), "XY": np.concatenate(XY),
           "O": np.concatenate(O), "RULE": np.concatenate(RULE),
           "Y": np.asarray(Y, np.float32), "sz": np.array(SZ), "TOK": np.array(TOK)}
    np.savez_compressed(CACHE, **out)
    print(f"wrote {CACHE}: {len(Y)} frames, {len(out['O'])} objects "
          f"({int(out['O'].sum())} hazardous; kinematic {int((out['RULE']==1).sum())}, "
          f"occlusion {int((out['RULE']==2).sum())})", flush=True)
    return out


def inv_ttc(V, XY):
    d = np.maximum(np.linalg.norm(XY, axis=1), 1e-3)
    closing = -(V * XY).sum(1) / d
    return np.where(closing > 0, closing / d, 0.0)


def fit_mahalanobis(Xtr, ytr):
    mus = [Xtr[ytr == c].mean(0) for c in (0, 1)]
    Xc = np.concatenate([Xtr[ytr == c] - mus[c] for c in (0, 1)])
    S = np.cov(Xc.T) + 1e-3 * np.eye(Xtr.shape[1])
    P = np.linalg.pinv(S)
    return lambda X: -np.min([np.einsum("ij,jk,ik->i", X - m, P, X - m) for m in mus], axis=0)


def fit_lda(Xtr, ytr):
    m0, m1 = Xtr[ytr == 0].mean(0), Xtr[ytr == 1].mean(0)
    Xc = np.concatenate([Xtr[ytr == 0] - m0, Xtr[ytr == 1] - m1])
    S = np.cov(Xc.T) + 1e-3 * np.eye(Xtr.shape[1])
    w = np.linalg.pinv(S) @ (m1 - m0)
    return lambda X: X @ w


def fit_knn(Xid, k=50, ref=20000, seed=0):
    r = np.random.default_rng(seed)
    if len(Xid) > ref:
        Xid = Xid[r.choice(len(Xid), ref, replace=False)]
    R = Xid / np.maximum(np.linalg.norm(Xid, axis=1, keepdims=True), 1e-9)
    def score(X):
        Q = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        out = np.empty(len(Q))
        for i in range(0, len(Q), 2048):
            sim = Q[i:i + 2048] @ R.T
            out[i:i + 2048] = -np.partition(-sim, k, axis=1)[:, k]
        return -out          # larger distance (smaller similarity) = more OOD
    return score


def frame_pool(s, sz):
    off = np.concatenate([[0], np.cumsum(sz)])
    return np.array([s[off[i]:off[i + 1]].max() for i in range(len(sz))])


def top1(s, O, sz):
    off = np.concatenate([[0], np.cumsum(sz)])
    hits = []
    for i in range(len(sz)):
        o = O[off[i]:off[i + 1]]
        if len(o) < 2 or o.sum() == 0:
            continue
        hits.append(o[int(np.argmax(s[off[i]:off[i + 1]]))])
    return 100 * np.mean(hits), len(hits)


def main():
    d = build() if not os.path.exists(CACHE) else dict(np.load(CACHE, allow_pickle=False))
    F, V, XY, O, RULE = d["F"], d["V"], d["XY"], d["O"], d["RULE"]
    Y, sz, TOK = d["Y"], d["sz"], [str(t) for t in d["TOK"]]
    off = np.concatenate([[0], np.cumsum(sz)])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\npool {len(Y)} frames, {len(O)} objects ({int(O.sum())} hazardous: "
          f"kinematic {int((RULE==1).sum())}, occlusion {int((RULE==2).sum())})")
    fold = scene_folds(TOK, Y, NFOLD)
    ofold = np.concatenate([np.full(sz[i], fold[i]) for i in range(len(sz))])

    scores = {"inverse TTC (no model)": inv_ttc(V, XY),
              "MSP-analogue": -F[:, -1]}
    for name in ("kNN k=50", "Mahalanobis", "LDA (supervision-matched)", "Object-level MLP (ours)"):
        scores[name] = np.zeros(len(O))

    for f in range(NFOLD):
        tri, tei = ofold != f, ofold == f
        Xtr, ytr = F[tri], O[tri]
        scores["kNN k=50"][tei] = fit_knn(Xtr[ytr == 0])(F[tei])
        scores["Mahalanobis"][tei] = fit_mahalanobis(Xtr, ytr)(F[tei])
        scores["LDA (supervision-matched)"][tei] = fit_lda(Xtr, ytr)(F[tei])
        fi = np.where(fold != f)[0]; fj = np.where(fold == f)[0]
        Ftr, Otr = group(F, sz, O, fi); Fte, Ote = group(F, sz, O, fj)
        _, so, _, _ = T.train_arm(1, (Ftr, Y[fi], Otr, None), (Fte, Y[fj], Ote, None),
                                  dev, 15, seed=0, quiet=True)
        scores["Object-level MLP (ours)"][tei] = so
        print(f"  fold {f} done", flush=True)

    print(f"\n{'method':30s}{'frame AUROC':>13}{'object AUROC':>14}{'top-1':>9}"
          f"{'  kin-AUROC':>12}{'  occ-AUROC':>12}")
    print("-" * 92)
    res = {}
    kin = (O == 0) | (RULE == 1)          # negatives + kinematic hazards only
    occ = (O == 0) | (RULE == 2)
    for name, s in scores.items():
        fa = 100 * T.auroc(frame_pool(s, sz), Y)
        oa = 100 * T.auroc(s, O)
        t1, n1 = top1(s, O, sz)
        ka = 100 * T.auroc(s[kin], O[kin]); oc = 100 * T.auroc(s[occ], O[occ])
        print(f"{name:30s}{fa:12.2f}%{oa:13.2f}%{t1:8.2f}%{ka:11.2f}%{oc:11.2f}%")
        res[name] = {"frame_auroc": fa, "object_auroc": oa, "top1": t1,
                     "kinematic_auroc": ka, "occlusion_auroc": oc}
    chance, n1 = top1(np.random.default_rng(0).random(len(O)), O, sz)
    print(f"{'chance (random ranking)':30s}{'':13}{50.0:13.2f}%{chance:8.2f}%")
    res["chance_top1"] = chance; res["n_top1_frames"] = n1

    print(f"\nours minus each external baseline (object AUROC), bootstrap over {len(O)} objects")
    ours = scores["Object-level MLP (ours)"]
    for name in ("inverse TTC (no model)", "MSP-analogue", "kNN k=50", "Mahalanobis",
                 "LDA (supervision-matched)"):
        lo, hi = boot_diff(ours, scores[name], O, lambda s, y: 100 * T.auroc(s, y), n=1500)
        pt = res["Object-level MLP (ours)"]["object_auroc"] - res[name]["object_auroc"]
        print(f"   vs {name:30s} {pt:+7.2f} pp  CI [{lo:+.2f},{hi:+.2f}]"
              f"{'  *' if lo > 0 else '   NOT SIGNIFICANT'}")
        res[f"vs_{name}"] = {"point": float(pt), "ci": [float(lo), float(hi)]}

    np.savez_compressed("results/cv_oof_scores.npz", ofold=ofold, O=O, sz=sz, Y=Y, **{k.replace(" ","_"): v for k, v in scores.items()})
    print("wrote results/cv_oof_scores.npz (out-of-fold scores + fold assignment)")
    json.dump(res, open("results/cv_external_baselines_rerun.json", "w"), indent=2)
    print("wrote results/cv_external_baselines_rerun.json")


if __name__ == "__main__":
    main()
