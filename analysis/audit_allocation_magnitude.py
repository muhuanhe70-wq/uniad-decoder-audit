"""Does the permuted-allocation control hold aggregate avoidance fixed?

WHY THIS EXISTS
---------------
allocation_control.py permutes the per-frame retention factor

    s_i = ||A^F_i|| / ||A^B_i||

and compares the real allocation against permuted ones. Its stated invariant is
that "the aggregate withdrawal is held fixed and only its allocation varies".
The multiset {s_i} is preserved by construction. The quantity the objection of
Li et al. is actually about, however, is the applied displacement

    m_i = s_i * ||A^B_i||

and permuting s while leaving ||A^B_i|| where it is preserves sum(m) only if s
and ||A^B|| are uncorrelated. This script measures whether they are, and
decomposes the reported advantage by whether the permuted allocation applied
more or less avoidance than the reference on each frame.

It needs no GPU and re-runs no scoring: the per-frame collision outcomes are
read back from the checkpoint allocation_control.py already wrote.

Usage (from the repository root of the UniAD working tree):

    python3 analysis/audit_allocation_magnitude.py \
        --box results/allocation_control_box.npz \
        --base work_dirs/baseline --arm work_dirs/quantfix --raw work_dirs/nocoloptim \
        --out results/allocation_magnitude_audit.json
"""
import argparse
import gc
import io
import json

import mmcv
import numpy as np
import torch

torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")

EPS = 1e-6


def load_traj(path):
    b = mmcv.load(path)["bbox_results"]
    tr = np.stack([f["planning_traj"].detach().cpu().numpy()[0]
                   for f in b]).astype(np.float64)
    del b
    gc.collect()
    return tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="results/allocation_control_box.npz")
    ap.add_argument("--base", default="work_dirs/R-0072_2026-08-17_baseline_full")
    ap.add_argument("--arm", default="work_dirs/R-0073_2026-08-17_quantfix_full")
    ap.add_argument("--raw", default="work_dirs/R-0084_2026-08-21_nocoloptim")
    ap.add_argument("--out", default="results/allocation_magnitude_audit.json")
    a = ap.parse_args()

    d = np.load(a.box)
    s, perms = d["s"], d["perms"]
    real, perm = d["real_alloc"], d["perm"]
    K = len(perms)

    B, F, R = (load_traj(p + "/results.pkl") for p in (a.base, a.arm, a.raw))
    nB = np.linalg.norm(B - R, axis=2).mean(1)
    nF = np.linalg.norm(F - R, axis=2).mean(1)
    moved_all = np.where(np.abs(F - B).max(axis=(1, 2)) > 1e-9)[0]
    moved = moved_all[nB[moved_all] > EPS]
    assert np.array_equal(moved, d["moved"]), "frame set differs from the run"
    nB, nF = nB[moved], nF[moved]

    # what each arm actually applies, per frame
    m_hat = s * nB                        # the reference Fhat; equals nF by construction
    m_perm = np.stack([s[perms[j]] * nB for j in range(K)])

    corr = float(np.corrcoef(s, nB)[0, 1])
    tot_hat = float(m_hat.sum())
    tot_perm = m_perm.sum(1)

    # decompose the advantage by whether the permuted arm avoided at least as much
    rows = []
    for j in range(K):
        ge = m_perm[j] >= m_hat - 1e-12
        dif = real - perm[j]
        rows.append({
            "draw": j,
            "total_applied": float(tot_perm[j]),
            "pct_of_reference": float(100 * tot_perm[j] / tot_hat),
            "n_ge": int(ge.sum()),
            "adv_ge": float(dif[ge].sum()),
            "informative_ge": int((dif[ge] != 0).sum()),
            "n_lt": int((~ge).sum()),
            "adv_lt": float(dif[~ge].sum()),
            "informative_lt": int((dif[~ge] != 0).sum()),
        })

    adv_ge = float(np.mean([r["adv_ge"] for r in rows]))
    adv_lt = float(np.mean([r["adv_lt"] for r in rows]))

    out = {
        "n_eligible": int(len(moved)),
        "k": int(K),
        "corr_retention_vs_published_magnitude": corr,
        "total_applied_reference": tot_hat,
        "total_applied_permuted_mean": float(tot_perm.mean()),
        "permuted_pct_of_reference_mean": float(100 * tot_perm.mean() / tot_hat),
        "per_draw": rows,
        "advantage_where_permuted_applied_at_least_as_much": adv_ge,
        "advantage_where_permuted_applied_less": adv_lt,
        "share_of_advantage_from_lower_magnitude_frames":
            float(adv_lt / (adv_ge + adv_lt)),
        "verdict": (
            "The permuted allocations do not hold aggregate applied avoidance "
            "fixed: they apply %.1f%% of the reference total, because retention "
            "correlates with published avoidance magnitude (r = %+.3f). The "
            "entire reported advantage arises on frames where the permuted "
            "allocation applied less avoidance, so this control does not "
            "separate allocation from magnitude."
            % (100 * tot_perm.mean() / tot_hat, corr)),
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)

    print("corr(s, ||A^B||)            %+.3f" % corr)
    print("reference total applied     %.2f" % tot_hat)
    print("permuted total, mean        %.2f  (%.1f%% of reference)"
          % (tot_perm.mean(), 100 * tot_perm.mean() / tot_hat))
    print("advantage, permuted >= ref  %+.2f  (informative %.1f/draw)"
          % (adv_ge, np.mean([r["informative_ge"] for r in rows])))
    print("advantage, permuted <  ref  %+.2f  (informative %.1f/draw)"
          % (adv_lt, np.mean([r["informative_lt"] for r in rows])))
    print("\n" + out["verdict"])
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
