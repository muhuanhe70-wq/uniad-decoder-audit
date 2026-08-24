"""Which public repositories carry the coordinate-truncation defect, not merely the file?

Finding `CollisionNonlinearOptimizer` or `occ_filter_range` in a repository only shows that it
vendored UniAD's planning stage. The defect itself lives in three consecutive lines of
planning_head.py, so each candidate file is fetched and inspected directly:

    pos_xy = torch.nonzero(occ_mask[0][cur_t], as_tuple=False)   # -> int64
    pos_xy = pos_xy[:, [1, 0]]
    pos_xy[:, 0] = (pos_xy[:, 0] - self.bev_h//2) * 0.5 + 0.25   # float written into int64

A repository is marked DEFECT when the nonzero call and the float-valued assignment are both
present and no cast to a floating dtype appears between them; FIXED when a cast is present;
ABSENT when the pattern is not there at all (a different planning stage, or a file that only
mentions the identifier).

The token is read from ~/.gh_token and is never printed, logged, or written to the output.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

TOKEN = open(os.path.expanduser("~/.gh_token")).read().strip()
HDRS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
        "User-Agent": "novelty-audit"}

NONZERO = re.compile(r"torch\.nonzero\s*\(\s*occ_mask")
ASSIGN = re.compile(r"pos_xy\s*\[\s*:\s*,\s*0\s*\]\s*=.*0\.5\s*\+\s*0\.25")
CAST = re.compile(r"pos_xy\s*=\s*pos_xy\s*\.\s*(to\s*\(\s*torch\.(float|double)|float\s*\(|double\s*\()")


def api(url, raw=False, tries=4):
    """GitHub occasionally drops the TLS connection mid-sweep; retry rather than lose the run."""
    for attempt in range(tries):
        r = urllib.request.Request(url, headers=HDRS)
        try:
            with urllib.request.urlopen(r, timeout=45) as h:
                b = h.read()
            return b.decode("utf-8", "replace") if raw else json.loads(b)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and attempt < tries - 1:
                time.sleep(20 * (attempt + 1))
                continue
            return {"_err": e.code}
        except Exception:
            if attempt < tries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            return {"_err": "network"}


def search_code(q, pages=2):
    items = []
    for p in range(1, pages + 1):
        u = (f"https://api.github.com/search/code?q={urllib.parse.quote(q)}"
             f"&per_page=100&page={p}")
        d = api(u)
        if "_err" in d:
            print(f"  search page {p}: HTTP {d['_err']}", flush=True)
            break
        items += d.get("items", [])
        if len(items) >= d.get("total_count", 0):
            break
        time.sleep(7)
    return items


def classify(text):
    if not NONZERO.search(text) or not ASSIGN.search(text):
        return "ABSENT"
    i = NONZERO.search(text).start()
    j = ASSIGN.search(text).start()
    if j < i:
        return "ABSENT"
    return "FIXED" if CAST.search(text[i:j]) else "DEFECT"


def main():
    items = search_code("occ_filter_range")
    # keep one candidate file per repository, preferring a planning_head-like path
    cand = {}
    for it in items:
        repo, path = it["repository"]["full_name"], it["path"]
        score = (2 if "planning_head" in path else 0) + (1 if path.endswith(".py") else 0)
        if repo not in cand or score > cand[repo][0]:
            cand[repo] = (score, path, it["url"])
    print(f"\n{len(items)} hits across {len(cand)} repositories; fetching each\n", flush=True)

    ckpt = "results/propagation_audit.partial.json"
    out = {"DEFECT": [], "FIXED": [], "ABSENT": [], "ERROR": []}
    done = set()
    if os.path.exists(ckpt):
        out = json.load(open(ckpt))
        done = {r["repo"] for v in out.values() for r in v}
        print(f"resuming: {len(done)} repositories already classified", flush=True)

    for n, (repo, (_, path, url)) in enumerate(sorted(cand.items()), 1):
        if repo in done:
            continue
        d = api(url)
        text = ""
        if isinstance(d, dict) and d.get("download_url"):
            text = api(d["download_url"], raw=True)
        verdict = "ERROR" if not isinstance(text, str) or not text else classify(text)
        out[verdict].append({"repo": repo, "path": path})
        print(f"  [{n:3d}/{len(cand)}] {verdict:7s} {repo}", flush=True)
        json.dump(out, open(ckpt, "w"), indent=2)
        time.sleep(0.4)

    print("\n" + "=" * 72)
    for k in ("DEFECT", "FIXED", "ABSENT", "ERROR"):
        print(f"{k:7s} {len(out[k]):3d}")
    print("=" * 72)
    print("\nrepositories carrying the defect:")
    for r in out["DEFECT"]:
        print(f"  {r['repo']:46s} {r['path']}")
    if out["FIXED"]:
        print("\nrepositories that cast before the assignment (already correct):")
        for r in out["FIXED"]:
            print(f"  {r['repo']:46s} {r['path']}")

    json.dump(out, open("results/propagation_audit.json", "w"), indent=2)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print("\nwrote results/propagation_audit.json")


if __name__ == "__main__":
    sys.exit(main())
