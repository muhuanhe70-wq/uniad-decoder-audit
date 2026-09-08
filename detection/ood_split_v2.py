"""Deterministic reconstruction of the OOD detector's train/test split.

WHY THIS EXISTS
---------------
`train_final_table3.py` splits the feature cache by **file mtime**
(`get_chronological_split`, lines 41-46). That was only ever a proxy for
"extraction order", and it has since been invalidated: until 2026-08-11,
`uniad_e2e.py` re-wrote `{token}.pt` on every forward pass of every eval run,
so any unrelated evaluation silently bumped mtimes and changed what "the 80/20
chronological split" meant. By 2026-08-11, 573 cache files had already been
rewritten this way (the write is now gated behind DUMP_OOD_FEATURES=1).

This module reconstructs the ORIGINAL split deterministically, from nuScenes
timestamps rather than from the filesystem, so it is stable no matter what
later runs do to mtimes.

HOW THE RECONSTRUCTION WORKS
----------------------------
`run_full_extraction.sh` ran the VAL split first, then the TRAIN split, each
through `extract_features_full.py` with shuffle=False. `NuScenesE2EDataset`
sorts its infos by `timestamp` (nuscenes_e2e_dataset.py:192). So extraction
order -- and therefore the original mtime order -- is:

    rank(token) = index in timestamp-sorted val infos                 if val
                  n_val + index in timestamp-sorted train infos       if train

The 80/20 split is then applied to this rank *separately per directory*
(normal / risk), exactly as the original per-directory mtime split did.

VALIDATION: `self_check()` asserts the reconstruction reproduces the original
n_train=27319 / n_test=6830, and `verify_against_log()` in
`ood_ema_ordering_study.py` re-scores the 5 saved checkpoints and requires
they reproduce the per-rep AUROCs recorded in
`logs_extraction/train_final_table3.log`. That is the real proof the
reconstruction is bit-exact, not just plausible.
"""
import json
import os
import pickle

INFO_VAL = 'data/infos/nuscenes_infos_temporal_val.pkl'
INFO_TRAIN = 'data/infos/nuscenes_infos_temporal_train.pkl'
NORMAL_DIR = './normal_features_v2'
RISK_DIR = './risk_features_v2'
TRAIN_RATIO = 0.8
PINNED_JSON = './ood_split_v2.json'


def _sorted_tokens(path):
    """Token order exactly as NuScenesE2EDataset yields it (sorted by timestamp)."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    infos = sorted(data['infos'], key=lambda e: e['timestamp'])
    return [i['token'] for i in infos]


def build_rank_map():
    """token -> extraction rank (val frames first, then train frames)."""
    val_tokens = _sorted_tokens(INFO_VAL)
    train_tokens = _sorted_tokens(INFO_TRAIN)
    rank = {t: i for i, t in enumerate(val_tokens)}
    off = len(val_tokens)
    for i, t in enumerate(train_tokens):
        rank[t] = off + i
    return rank, len(val_tokens)


def _dir_tokens(directory):
    return [f[:-3] for f in os.listdir(directory) if f.endswith('.pt')]


def get_split(train_ratio=TRAIN_RATIO):
    """Returns (normal_train, normal_test, risk_train, risk_test) token lists.

    Ordered by extraction rank. Never touches mtime.
    """
    rank, _ = build_rank_map()
    out = []
    for d in (NORMAL_DIR, RISK_DIR):
        toks = _dir_tokens(d)
        missing = [t for t in toks if t not in rank]
        if missing:
            raise RuntimeError(
                f'{len(missing)} cached tokens in {d} are absent from the nuScenes '
                f'infos (e.g. {missing[:3]}) -- cannot rank them deterministically.')
        toks.sort(key=lambda t: rank[t])
        cut = int(len(toks) * train_ratio)
        out.extend([toks[:cut], toks[cut:]])
    return out[0], out[1], out[2], out[3]


def feature_path(token):
    """Cache path for a token (risk dir takes precedence, as in extraction)."""
    p = os.path.join(RISK_DIR, token + '.pt')
    return p if os.path.exists(p) else os.path.join(NORMAL_DIR, token + '.pt')


def self_check(verbose=True):
    """Assert the reconstruction matches the original run's recorded split sizes."""
    n_tr, n_te, r_tr, r_te = get_split()
    train_n, test_n = len(n_tr) + len(r_tr), len(n_te) + len(r_te)
    rank, n_val = build_rank_map()

    assert train_n == 27319, f'n_train={train_n}, expected 27319 (see train_final_table3.log)'
    assert test_n == 6830, f'n_test={test_n}, expected 6830 (see train_final_table3.log)'
    assert not (set(n_tr) | set(r_tr)) & (set(n_te) | set(r_te)), 'train/test overlap'
    assert not set(n_tr) & set(r_tr), 'a token is both normal and risk'

    # The test split should consist of train-stage tokens only. This matters:
    # val-split evals rewrite their own tokens' features, so if any test token
    # were a val token its FEATURE CONTENT (not just mtime) could have drifted
    # since the checkpoints were trained.
    test_tokens = set(n_te) | set(r_te)
    val_leak = {t for t in test_tokens if rank[t] < n_val}
    assert not val_leak, (
        f'{len(val_leak)} test tokens are val-stage frames whose cached features '
        f'may have been overwritten by later evals -- results would be unsound.')

    if verbose:
        print(f'split OK: n_train={train_n} n_test={test_n} '
              f'(normal {len(n_tr)}/{len(n_te)}, risk {len(r_tr)}/{len(r_te)}); '
              f'0 val-stage tokens in test')
    return n_tr, n_te, r_tr, r_te


def pin(path=PINNED_JSON):
    """Write the split to disk so it is reproducible with no nuScenes infos present."""
    n_tr, n_te, r_tr, r_te = self_check(verbose=False)
    rank, _ = build_rank_map()
    rows = {}
    for toks, split, label in ((n_tr, 'train', 0), (n_te, 'test', 0),
                               (r_tr, 'train', 1), (r_te, 'test', 1)):
        for t in toks:
            rows[t] = {'split': split, 'label': label, 'rank': rank[t]}
    with open(path, 'w') as f:
        json.dump(rows, f)
    print(f'pinned {len(rows)} tokens -> {path}')
    return rows


if __name__ == '__main__':
    self_check()
    pin()
