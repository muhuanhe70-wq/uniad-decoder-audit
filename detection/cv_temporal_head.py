"""Scene-disjoint 5-fold cross-validation of the temporal head against its per-frame control.

WHY THIS EXISTS, AND THE RULE IT COMMITS TO IN ADVANCE
------------------------------------------------------
On the original chronological split the temporal arm was consistent in sign but not
resolvable: frame AUROC +0.82 +/- 0.59 pp with 5/5 seeds positive, recall@9.25 %
+2.84 +/- 2.29 pp with 4/5 positive, and frame-level bootstrap CIs that both include zero
([-0.10, +0.86] and [-1.51, +6.42]). The limiting factor is the held-out slice: 4,571 frames
of which only 804 are risk frames. That cannot resolve a ~1 pp effect.

Changing the protocol after seeing a null is p-hacking unless the rules are fixed first, so:

  1. Both arms use the IDENTICAL folds and the identical object set. Only the input width
     differs (257 vs 773).
  2. BOTH protocols get reported. The chronological-split result above is not withdrawn or
     replaced by whatever this produces; it is reported next to it.
  3. If the pooled out-of-fold CI still includes zero, the temporal head is dropped. No
     third protocol, no threshold search, no metric substitution.

Folds are disjoint BY SCENE, not by frame. Frames inside a nuScenes scene are 0.5 s apart and
massively correlated; a random frame-level fold would put a frame's own neighbours in the
training set and inflate both arms. Scene-disjoint folding is also strictly harder than the
chronological split, which cuts mid-scene.

The pool is the clean training set plus the held-out slice -- 100 % val-frame-free, so this
composes with the innovation-1 numbers rather than quietly reintroducing the contamination.
"""
import argparse
import json
import os

import mmcv
import numpy as np
import torch

import ood_split_v2 as sm
import train_temporal_head as T

CACHE = "cache_temporal_cv.npz"
FPRS = [0.05, 0.0925, 0.15, 0.20, 0.30]
NFOLD = 5


def build():
    ntr, nte, rtr, rte = sm.get_split()
    import gc
    b = mmcv.load("work_dirs/R-0006_2026-08-04_baseline_pf_a/results.pkl")["bbox_results"]
    val = set(f["token"] for f in b); del b; gc.collect()

    toks = ([t for t in ntr if t not in val] + [t for t in nte if t not in val]
            + [t for t in rtr if t not in val] + [t for t in rte if t not in val])
    labs = ([0.0] * len([t for t in ntr if t not in val] + [t for t in nte if t not in val])
            + [1.0] * len([t for t in rtr if t not in val] + [t for t in rte if t not in val]))
    print(f"pool: {len(toks)} val-free frames ({int(sum(labs))} risk)", flush=True)

    cached = set(ntr) | set(nte) | set(rtr) | set(rte)
    prev = T.build_prev_map(cached)
    print("loading feature cache ...", flush=True)
    cache = T.load_cache(sorted(cached))
    lab = json.load(open("risk_objects_v3.json"))

    keep = set(T.assemble(toks, labs, cache, prev, 3, lab)[3])
    out = {}
    ref_tok = None
    for K in (1, 3):
        F, Y, O, TOK = T.assemble(toks, labs, cache, prev, K, lab)
        idx = [i for i, t in enumerate(TOK) if t in keep]
        F = [F[i] for i in idx]; O = [O[i] for i in idx]
        Y = Y[idx]; TOK = [TOK[i] for i in idx]
        if ref_tok is None:
            ref_tok = TOK
        assert TOK == ref_tok, "arms disagree on the frame set"
        out[f"K{K}_F"] = np.concatenate(F)
        out[f"K{K}_sz"] = np.array([len(f) for f in F])
        out[f"K{K}_O"] = np.concatenate(O)
    out["Y"] = np.asarray(ref_tok and Y, np.float32)
    out["TOK"] = np.array(ref_tok)
    np.savez_compressed(CACHE, **out)
    print(f"wrote {CACHE}  ({len(ref_tok)} frames)", flush=True)
    return dict(np.load(CACHE, allow_pickle=False))


def scene_folds(tokens, y, nfold, seed=0):
    """Assign whole scenes to folds, greedily balancing the risk-frame count."""
    infos = {}
    for f in ("data/infos/nuscenes_infos_temporal_train.pkl",
              "data/infos/nuscenes_infos_temporal_val.pkl"):
        for i in mmcv.load(f)["infos"]:
            infos[i["token"]] = i["scene_token"]
    scenes = {}
    for i, t in enumerate(tokens):
        scenes.setdefault(infos[t], []).append(i)
    order = sorted(scenes, key=lambda s: -sum(y[j] for j in scenes[s]))
    load = np.zeros(nfold); fold = np.full(len(tokens), -1)
    for s in order:
        f = int(np.argmin(load))
        for j in scenes[s]:
            fold[j] = f
        load[f] += sum(y[j] for j in scenes[s])
    print(f"  {len(scenes)} scenes -> {nfold} folds; risk frames per fold: "
          + ", ".join(str(int(x)) for x in load))
    assert (fold >= 0).all()
    return fold


def group(F, sz, O, idx):
    off = np.concatenate([[0], np.cumsum(sz)])
    Fl = [F[off[i]:off[i + 1]] for i in idx]
    Ol = [O[off[i]:off[i + 1]] for i in idx]
    return Fl, Ol


def recall_at(s, y, f):
    return float(100 * (s[y == 1] > np.quantile(s[y == 0], 1 - f)).mean())


def boot_diff(a, b, y, fn, n=4000, seed=0):
    r = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if y[i].sum() < 2 or (1 - y[i]).sum() < 2:
            continue
        d.append(fn(a[i], y[i]) - fn(b[i], y[i]))
    return np.percentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    d = build() if (a.rebuild or not os.path.exists(CACHE)) else dict(np.load(CACHE, allow_pickle=False))
    Y = d["Y"]; TOK = [str(t) for t in d["TOK"]]
    print(f"\npooled frames {len(Y)} ({int(Y.sum())} risk)")
    fold = scene_folds(TOK, Y, NFOLD)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_seed = {1: [], 3: []}
    oof_last = {}
    for s in a.seeds:
        oof = {K: np.zeros(len(Y)) for K in (1, 3)}
        for f in range(NFOLD):
            tr_i = np.where(fold != f)[0]; te_i = np.where(fold == f)[0]
            for K in (1, 3):
                F, sz, O = d[f"K{K}_F"], d[f"K{K}_sz"], d[f"K{K}_O"]
                Ftr, Otr = group(F, sz, O, tr_i)
                Fte, Ote = group(F, sz, O, te_i)
                _, _, fs, _ = T.train_arm(K, (Ftr, Y[tr_i], Otr, None),
                                          (Fte, Y[te_i], Ote, None), dev, a.epochs,
                                          seed=s, quiet=True)
                oof[K][te_i] = fs
            print(f"  seed {s} fold {f}: done", flush=True)
        row = {}
        for K in (1, 3):
            row[K] = {"frame_auroc": 100 * T.auroc(oof[K], Y)}
            for fp in FPRS:
                row[K][f"recall@{fp}"] = recall_at(oof[K], Y, fp)
            per_seed[K].append(row[K])
        oof_last = oof
        print(f"  seed {s}: frame AUROC K1 {row[1]['frame_auroc']:.2f}%  "
              f"K3 {row[3]['frame_auroc']:.2f}%   recall@9.25% "
              f"{row[1]['recall@0.0925']:.2f} -> {row[3]['recall@0.0925']:.2f}", flush=True)

    print(f"\npooled out-of-fold, {len(Y)} frames ({int(Y.sum())} risk)")
    print(f"{'metric':18s}{'K=1 (control)':>20}{'K=3 (temporal)':>20}{'paired diff':>20}")
    print("-" * 78)
    res = {}
    for k in per_seed[1][0]:
        v1 = np.array([r[k] for r in per_seed[1]]); v3 = np.array([r[k] for r in per_seed[3]])
        dd = v3 - v1
        print(f"{k:18s}{v1.mean():13.2f} ±{v1.std():5.2f}{v3.mean():13.2f} ±{v3.std():5.2f}"
              f"{dd.mean():+13.2f} ±{dd.std():4.2f} {int((dd>0).sum())}/{len(dd)}")
        res[k] = {"K1": [float(v1.mean()), float(v1.std())],
                  "K3": [float(v3.mean()), float(v3.std())],
                  "diff": float(dd.mean()), "seeds_positive": int((dd > 0).sum())}

    print("\nframe-level bootstrap on the pooled out-of-fold scores (last seed) "
          "-- THIS IS THE DECIDING NUMBER")
    for lab, fn in (("frame AUROC", lambda s, y: 100 * T.auroc(s, y)),
                    ("recall@9.25%", lambda s, y: recall_at(s, y, 0.0925)),
                    ("recall@15%", lambda s, y: recall_at(s, y, 0.15))):
        pt = fn(oof_last[3], Y) - fn(oof_last[1], Y)
        lo, hi = boot_diff(oof_last[3], oof_last[1], Y, fn)
        sig = "*  SIGNIFICANT" if (lo > 0 or hi < 0) else "   n.s. -> DROP per the pre-set rule"
        print(f"   {lab:14s} {pt:+7.2f} pp   CI [{lo:+.2f},{hi:+.2f}] {sig}")
        res[f"boot_{lab}"] = {"point": float(pt), "ci": [float(lo), float(hi)]}

    json.dump(res, open("results/temporal_head_cv.json", "w"), indent=2)
    print("\nwrote results/temporal_head_cv.json")


if __name__ == "__main__":
    main()
