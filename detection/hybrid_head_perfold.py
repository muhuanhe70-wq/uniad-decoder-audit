"""Does feeding the rule's own outputs to the learned head beat either alone?

Established: on kinematic hazards the rule computed from perception scores 96.38 % object
AUROC against the learned head's 96.78 % -- a tie, so the head adds nothing there. On
occlusion hazards the head leads 85.70 % to 76.44 %, because the rule's visibility gate has no
perception counterpart. And the two rules do not compose: max-combining them drops kinematic
from 96.38 % to 66.96 %, since a nearby barrier outranks a genuine closing vehicle.

That is an argument for a hybrid rather than for either component, and it is testable: append
the rule's intermediate quantities to the 257-d features and retrain identically. If the head
already knows everything the rule knows, this changes nothing. If it improves, the rule
carries information the features do not expose, and the combination is a design result rather
than a bigger model.

Arms, on identical folds/frames/seeds, differing only in input width:
  A  257-d track features                                    (the current head)
  B  257-d + 9 rule quantities (TTC, closing, dist, lateral, forward, class flags, rule hits)
  C  the 9 rule quantities alone                             (is the rule all that matters?)
"""
import json
import numpy as np, torch, mmcv, os
import train_temporal_head as T
from cv_temporal_head import NFOLD, scene_folds, group, boot_diff
from rule_replica_baseline import (ego_speed, KIN_CLASSES, OCC_CLASSES, LABELS,
                                   MAX_DISTANCE_KINEMATIC, LATERAL_GATE,
                                   TTC_THRESHOLD, MIN_CLOSING_SPEED, MAX_DISTANCE_OCC)

d = dict(np.load("cache_ext_baselines.npz", allow_pickle=False))
F, V, XY, O, RULE, sz = d["F"], d["V"], d["XY"], d["O"], d["RULE"], d["sz"]
Y = d["Y"]; TOK = [str(t) for t in d["TOK"]]
L = np.load(LABELS)["L"]
off = np.concatenate([[0], np.cumsum(sz)])
es = ego_speed(TOK); eso = np.concatenate([np.full(sz[i], es[i]) for i in range(len(sz))])
lat, fwd = XY[:, 0], XY[:, 1]
dist = np.maximum(np.linalg.norm(XY, axis=1), 1e-3)
rel = V.copy(); rel[:, 1] -= eso
closing = -(rel * XY).sum(1) / dist
with np.errstate(divide="ignore", invalid="ignore"):
    ttc = np.where(closing > MIN_CLOSING_SPEED, dist / np.maximum(closing, 1e-6), 1e3)
is_kin = np.isin(L, list(KIN_CLASSES)); is_occ = np.isin(L, list(OCC_CLASSES))
kin_hit = (is_kin & (dist <= MAX_DISTANCE_KINEMATIC) & (fwd > 0) & (np.abs(lat) < LATERAL_GATE)
           & (closing > MIN_CLOSING_SPEED) & (ttc < TTC_THRESHOLD))
occ_hit = is_occ & (dist <= MAX_DISTANCE_OCC)
RF = np.stack([np.clip(1.0/np.maximum(ttc,1e-3),0,10), np.clip(closing,-20,20),
               np.clip(dist,0,60), np.clip(lat,-40,40), np.clip(fwd,-40,40),
               is_kin.astype(float), is_occ.astype(float),
               kin_hit.astype(float), occ_hit.astype(float)], 1).astype(np.float32)
print(f"rule feature block: {RF.shape}")

fold = scene_folds(TOK, Y, NFOLD)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARMS = {"A 257-d features (current)": F,
        "B 257-d + rule features": np.concatenate([F, RF], 1),
        "C rule features only": RF}
kinm = (O == 0) | (RULE == 1); occm = (O == 0) | (RULE == 2)
oof = {k: np.zeros(len(O)) for k in ARMS}
for f in range(NFOLD):
    fi = np.where(fold != f)[0]; fj = np.where(fold == f)[0]
    ofold_te = np.concatenate([np.full(sz[i], fold[i]) for i in range(len(sz))]) == f
    for k, X in ARMS.items():
        Ftr, Otr = group(X, sz, O, fi); Fte, Ote = group(X, sz, O, fj)
        _, so, _, _ = T.train_arm(1, (Ftr, Y[fi], Otr, None), (Fte, Y[fj], Ote, None),
                                  dev, 15, seed=0, quiet=True)
        oof[k][ofold_te] = so
    print(f"  fold {f} done", flush=True)

def top1(s):
    h = []
    for i in range(len(sz)):
        o = O[off[i]:off[i+1]]
        if len(o) < 2 or o.sum() == 0: continue
        h.append(o[int(np.argmax(s[off[i]:off[i+1]]))])
    return 100*np.mean(h)
def fp(s):
    return np.array([s[off[i]:off[i+1]].max() for i in range(len(sz))])

print(f"\n{'arm':30s}{'frame':>9}{'object':>9}{'top-1':>8}{'kin':>8}{'occ':>8}")
res = {}
for k, s in oof.items():
    r = (100*T.auroc(fp(s),Y), 100*T.auroc(s,O), top1(s),
         100*T.auroc(s[kinm],O[kinm]), 100*T.auroc(s[occm],O[occm]))
    print(f"{k:30s}{r[0]:8.2f}%{r[1]:8.2f}%{r[2]:7.2f}%{r[3]:7.2f}%{r[4]:7.2f}%")
    res[k] = list(r)
print(f"{'rule replica, best arm':30s}{'':9}{70.45:8.2f}%{72.09:7.2f}%{96.38:7.2f}%{76.44:7.2f}%")
print("\nB minus A (the hybrid question), bootstrap over objects")
for lab, fn in (("object AUROC", lambda s,y: 100*T.auroc(s,y)),):
    lo,hi = boot_diff(oof["B 257-d + rule features"], oof["A 257-d features (current)"], O, fn, n=1500)
    pt = res["B 257-d + rule features"][1] - res["A 257-d features (current)"][1]
    print(f"   {lab:14s} {pt:+7.2f} pp  CI [{lo:+.2f},{hi:+.2f}]"
          f"{'  *  HYBRID WINS' if lo>0 else '   n.s. -> the head already knows the rule'}")
ofold_full = np.concatenate([np.full(sz[i], fold[i]) for i in range(len(sz))])
np.savez_compressed("results/hybrid_oof_scores.npz", ofold=ofold_full, O=O, sz=sz,
                    Y=Y, RULE=RULE,
                    **{k.split()[0]: v for k, v in oof.items()})
print("wrote results/hybrid_oof_scores.npz (per-arm out-of-fold scores + folds)")
json.dump(res, open("results/hybrid_head_rerun.json","w"), indent=2)
print("\nwrote results/hybrid_head_rerun.json")
