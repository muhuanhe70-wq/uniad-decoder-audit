"""Innovation 2: a temporal object-level OOD head, and its per-frame control.

MOTIVATION, AND ONE MOTIVATION THAT DIED
----------------------------------------
The risk labels are defined on SEQUENCES, not frames: `generate_object_level_risk.py` keeps
a kinematic hazard only if it persists >= 2 consecutive frames and an occlusion hazard only
if it persists >= 3. The model that consumes those labels is per-frame. Aligning the model's
receptive field with the label's definition is the whole idea. Occlusion is 37,170 of the
54,034 hazardous objects (69 %), and "was visible, now is not" is inherently temporal.

The motivation that died, recorded so nobody re-derives it: the `reversal` rule compares
box_velocity at t against t-1 and so is unreachable by a per-frame model -- but it fires on
exactly 2 objects out of 16,864 kinematic ones. It cannot carry any result. Checked before
building anything on it.

THE TRAP THIS GUARDS AGAINST
----------------------------
A previously published 82.25 % AUROC on this project was RETRACTED: it was an artefact of
block-ordered test frames under EMA smoothing -- "temporal" smoothing across frames that were
not temporally adjacent. A temporal model is the same trap with more parameters. Guards:

  * history is resolved through the nuScenes `prev` pointer within one scene, never by array
    order. A frame whose predecessor is not cached gets a zero vector and a validity flag.
  * the temporal arm and the per-frame control are trained and evaluated on the IDENTICAL
    object set (only objects whose frame has the required history depth), so the comparison
    cannot be moved by changing who is scored.
  * causal only: frames t-1 .. t-K+1. No future frame is ever read.
  * val-frame-free training set, identical to train_clean_heads.py, so the numbers compose
    with the innovation-1 result.

WHAT IS COMPARED
----------------
Both arms are object-level (innovation 1's supervision) on the same clean split. Arm K=1 is
innovation 1 exactly. Arm K=3 adds two frames of causal history as feature deltas. Reported
on the untouched held-out slice: object AUROC, frame AUROC, and risk-frame recall at the
deployed normal-frame trigger rate.
"""
import argparse
import json
import os

import mmcv
import numpy as np
import torch
import torch.nn as nn

import ood_split_v2 as sm
from ood_spatial_gate import spatial_gate

FLIP = np.array([-1.0, 1.0])
MATCH_R = 2.0
ETA_DEPLOYED = -23.0
SEED = 0
CACHE = "cache_temporal_objects.npz"


class OOD_Energy_MLP(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(dropout * 0.7),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def auroc(s, y):
    o = np.argsort(s, kind="mergesort"); s, y = s[o], y[o]
    r = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[i:j + 1] = 0.5 * (i + j) + 1.0; i = j + 1
    p, n = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n)) if p and n else float("nan")


def build_prev_map(cached):
    """token -> immediate cached predecessor in the SAME scene, via the nuScenes pointer."""
    infos = {}
    for f in ("data/infos/nuscenes_infos_temporal_train.pkl",
              "data/infos/nuscenes_infos_temporal_val.pkl"):
        for i in mmcv.load(f)["infos"]:
            infos[i["token"]] = i
    prev = {}
    for t in cached:
        i = infos.get(t)
        if i is None:
            continue
        p = i.get("prev", "")
        if p and p in cached and p in infos and infos[p]["scene_token"] == i["scene_token"]:
            prev[t] = p
    return prev


def load_cache(tokens):
    """token -> (feats (n,257) ungated, track_ids (n,), xy (n,2), gate (n,))."""
    out = {}
    for t in tokens:
        try:
            d = torch.load(sm.feature_path(t), map_location="cpu")
        except Exception:
            continue
        q = d["track_query"].to(torch.float32)
        if q.shape[0] == 0:
            continue
        xy = d["boxes_2d_xy"].to(torch.float32).numpy().astype(np.float64)
        f = torch.cat([q, d["logits"].to(torch.float32)], -1).numpy().astype(np.float32)
        out[t] = (f, d["track_ids"].numpy(), xy, spatial_gate(xy))
    return out


def assemble(tokens, labels, cache, prev, K, obj_labels):
    """Per-frame arrays of temporal object features and object labels.

    Feature layout: [f_t, f_t - f_{t-1}, ..., f_t - f_{t-K+1}, validity flags].
    A missing history frame or a track absent at t-k contributes a zero delta and flag 0.
    """
    F, Y, O, TOK = [], [], [], []
    for t, fl in zip(tokens, labels):
        c = cache.get(t)
        if c is None:
            continue
        f, tid, xy, g = c
        if not g.any():
            continue
        # require the full causal chain to exist, so every scored object is comparable
        chain, cur = [], t
        ok = True
        for _ in range(K - 1):
            p = prev.get(cur)
            if p is None or p not in cache:
                ok = False; break
            chain.append(p); cur = p
        if not ok:
            continue
        base = f[g]; ids = tid[g]
        parts = [base]
        flags = []
        for p in chain:
            pf, ptid, _, _ = cache[p]
            pos = {int(k): j for j, k in enumerate(ptid)}
            idx = np.array([pos.get(int(k), -1) for k in ids])
            hit = idx >= 0
            hist = np.zeros_like(base)
            hist[hit] = pf[idx[hit]]
            delta = np.where(hit[:, None], base - hist, 0.0).astype(np.float32)
            parts.append(delta)
            flags.append(hit.astype(np.float32))
        feats = np.concatenate(parts + [np.stack(flags, 1)] if flags else parts, 1) \
            if flags else np.concatenate(parts, 1)
        ol = np.zeros(len(base), np.float32)
        rec = obj_labels.get(t)
        if rec:
            pts = np.array([np.array(o["ego_xy_lat_fwd"]) * FLIP for o in rec["objects"]])
            if len(pts):
                dd = np.linalg.norm(xy[g][:, None, :] - pts[None, :, :], axis=-1)
                ol = (dd.min(1) < MATCH_R).astype(np.float32)
        F.append(feats.astype(np.float32)); Y.append(fl); O.append(ol); TOK.append(t)
    return F, np.array(Y, np.float32), O, TOK


def frame_scores(model, F, dev, obj_level=True):
    out = []
    with torch.no_grad():
        for f in F:
            s = model(torch.tensor(f, device=dev)).squeeze(-1)
            out.append(float(s.max().item()))
    return np.array(out)


def train_arm(K, tr, te, dev, epochs=15, seed=SEED, quiet=False):
    Ftr, Ytr, Otr, _ = tr
    Fte, Yte, Ote, _ = te
    Xo = np.concatenate(Ftr); Yo = np.concatenate(Otr)
    Xe = np.concatenate(Fte); Ye = np.concatenate(Ote)
    dim = Xo.shape[1]
    if not quiet:
        print(f"    input dim {dim}   train objects {len(Xo)} ({int(Yo.sum())} hazardous, "
              f"{100*Yo.mean():.2f}%)   test objects {len(Xe)}", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)
    m = OOD_Energy_MLP(dim).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    pw = torch.tensor((len(Yo) - Yo.sum()) / max(Yo.sum(), 1), device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    X = torch.tensor(Xo, device=dev); Yt = torch.tensor(Yo, device=dev)
    Xt = torch.tensor(Xe, device=dev)
    n = len(X); ix = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(ix); tot = 0.0
        m.train()
        for k in range(0, n, 4096):
            sub = torch.tensor(ix[k:k + 4096], device=dev)
            l = lossf(m(X[sub]).squeeze(-1), Yt[sub])
            opt.zero_grad(); l.backward(); opt.step(); tot += float(l)
        if (ep % 5 == 0 or ep == epochs - 1) and not quiet:
            m.eval()
            with torch.no_grad():
                so = m(Xt).squeeze(-1).cpu().numpy()
            print(f"      epoch {ep:2d} loss {tot/max(n//4096,1):.4f}  "
                  f"object AUROC {100*auroc(so, Ye):.2f}%", flush=True)
    m.eval()
    with torch.no_grad():
        so = m(Xt).squeeze(-1).cpu().numpy()
    off = 0; fs = []
    for f in Fte:
        fs.append(so[off:off + len(f)].max()); off += len(f)
    return m, so, np.array(fs), Ye


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ntr, nte, rtr, rte = sm.get_split()
    import gc
    b = mmcv.load("work_dirs/R-0006_2026-08-04_baseline_pf_a/results.pkl")["bbox_results"]
    val = set(f["token"] for f in b); del b; gc.collect()

    tr_tok = [t for t in ntr if t not in val] + [t for t in rtr if t not in val]
    tr_lab = [0.0] * len([t for t in ntr if t not in val]) + [1.0] * len([t for t in rtr if t not in val])
    te_tok = list(nte) + list(rte)
    te_lab = [0.0] * len(nte) + [1.0] * len(rte)
    print(f"clean train {len(tr_tok)} frames | held-out test {len(te_tok)} frames", flush=True)

    cached = set(ntr) | set(nte) | set(rtr) | set(rte)
    print("building causal prev map ...", flush=True)
    prev = build_prev_map(cached)
    print(f"  {len(prev)} of {len(cached)} cached frames have a cached same-scene predecessor",
          flush=True)

    print("loading feature cache (this is the slow part) ...", flush=True)
    cache = load_cache(sorted(cached))
    print(f"  {len(cache)} frames usable", flush=True)

    obj_labels = json.load(open("risk_objects_v3.json"))
    Kmax = max(a.K)

    results = {}
    scores = {}
    for K in a.K:
        print(f"\n[K={K}] {'per-frame control (= innovation 1)' if K == 1 else f'temporal, {K} frames of causal history'}")
        # NOTE: assembled with Kmax's frame filter so every arm scores the SAME objects
        tr = assemble(tr_tok, tr_lab, cache, prev, K, obj_labels)
        te = assemble(te_tok, te_lab, cache, prev, K, obj_labels)
        # restrict to the frames that the deepest arm can also serve
        if K < Kmax:
            keep_tr = set(assemble(tr_tok, tr_lab, cache, prev, Kmax, obj_labels)[3])
            keep_te = set(assemble(te_tok, te_lab, cache, prev, Kmax, obj_labels)[3])
            def filt(x, keep):
                F, Y, O, T = x
                idx = [i for i, t in enumerate(T) if t in keep]
                return ([F[i] for i in idx], Y[idx], [O[i] for i in idx], [T[i] for i in idx])
            tr = filt(tr, keep_tr); te = filt(te, keep_te)
        print(f"    frames: train {len(tr[0])}, test {len(te[0])}")
        m, so, fs, Ye = train_arm(K, tr, te, dev, a.epochs)
        Yte = te[1]
        results[f"K{K}"] = {"object_auroc": 100 * auroc(so, Ye),
                            "frame_auroc": 100 * auroc(fs, Yte),
                            "n_test_frames": len(te[0]), "n_test_objects": len(so)}
        scores[f"K{K}"] = (fs, Yte)
        torch.save(m.state_dict(), f"ood_head_temporal_K{K}.pth")
        print(f"    -> object AUROC {results[f'K{K}']['object_auroc']:.2f}%   "
              f"frame AUROC {results[f'K{K}']['frame_auroc']:.2f}%")

    # recall at a matched normal-frame trigger rate, the deployed operating point
    print("\n[operating point] risk-frame recall at a matched normal-frame trigger rate")
    base_fs, base_y = scores[f"K{a.K[0]}"]
    target = 0.0925          # deployed head's held-out normal-frame trigger rate
    for K in a.K:
        fs, y = scores[f"K{K}"]
        thr = float(np.quantile(fs[y == 0], 1 - target))
        rec = float(100 * (fs[y == 1] > thr).mean())
        results[f"K{K}"]["recall_at_matched_fpr"] = rec
        print(f"    K={K}: eta {thr:8.3f}   risk-frame recall {rec:5.2f}%")

    json.dump(results, open("results/temporal_head.json", "w"), indent=2)
    print("\nwrote results/temporal_head.json")
    if len(a.K) > 1:
        d_obj = results[f"K{a.K[-1]}"]["object_auroc"] - results[f"K{a.K[0]}"]["object_auroc"]
        d_rec = results[f"K{a.K[-1]}"]["recall_at_matched_fpr"] - results[f"K{a.K[0]}"]["recall_at_matched_fpr"]
        print(f"\nTEMPORAL GAIN over the per-frame control on the identical object set:")
        print(f"    object AUROC {d_obj:+.2f} pp    recall@matched FPR {d_rec:+.2f} pp")


if __name__ == "__main__":
    main()
