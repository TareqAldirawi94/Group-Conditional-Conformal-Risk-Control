#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================================
# ddr_crc_eval.py
# ---------------------------------------------------------------------------
# Group-conditional Conformal Risk Control evaluation for the DDR example
# (Section 5). Independent reference implementation, faithful to:
#   - shaped slice           C_{j,t}(x) = {k in G_j : s(x,k) - b(x,k) <= t},  b=0
#   - per-group threshold     t_j = inf{ t : (sum_{cal, y in G_j} loss + B)/(n_j+1) <= alpha },
#                             t_j = lambda_max (FALLBACK) if n_j+1 < B/alpha or infeasible
#   - deployed union          C_PW(x) = { k : s(x,k) <= max_{j: k in G_j} t_j }
#   - reconciliation          Delta_j = R_j^slice - R_j^dep  (>= 0),  bound <= B*DeltaS_j
# Score: randomized APS  s(x,k) = sum_{p_j>p_k} p_j + U_x * p_k.
# U_MODE=redraw (default): new independent U_x per image and split; one U_x shared
# across candidate labels AND all methods within that redraw.
# U_MODE=fixed: sensitivity mode, one U_x bound to image ID for all redraws.
# Loss : ordinal        l(C,y) = min_{v in C}|v-y| if C nonempty else B (=4).
#
# Input : scores_ddr_pool.csv  (id, grade, p0..p4). Missing -> FAIL unless
# explicitly DDR_ALLOW_SIMULATED=1 for code testing.
# Output: ddr_crc_summary.csv, ddr_crc_checks.txt  (+ printed report)
# ===========================================================================
import os, sys, json, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd

ALPHA   = 0.10
B       = 4.0
K       = 5
N_REDRAW= int(os.environ.get("DDR_REDRAWS", "100"))
if N_REDRAW < 2:
    raise ValueError("DDR_REDRAWS must be at least 2")
SEED    = 4100
U_MODE  = os.environ.get("DDR_U_MODE", "redraw").strip().lower()
if U_MODE not in ("redraw", "fixed"):
    raise ValueError("DDR_U_MODE must be redraw (primary) or fixed (sensitivity)")
LAMBDA_MAX = 1.0 + 1e-9          # admits every label (APS scores lie in [0,1])
CSV_IN  = os.environ.get("DDR_SCORES", "scores_ddr_pool.csv")
OUTDIR  = os.environ.get("DDR_CRC_OUT", ".")

DIST = np.abs(np.arange(K)[:, None] - np.arange(K)[None, :]).astype(float)  # |u-v|

# --------------------------- APS score -------------------------------------
def aps_scores(P, U):
    """Randomized APS: s[i,k] = sum_{p_j>p_k} p_j + U[i]*p_k."""
    N = P.shape[0]; S = np.empty_like(P)
    for i in range(N):
        p = P[i]
        greater = (p[None, :] > p[:, None])          # greater[k,j] = p_j > p_k
        S[i] = (greater * p[None, :]).sum(1) + U[i] * p
    return S

# --------------------------- loss ------------------------------------------
def losses_of_sets(member, y):
    """member: (n,K) bool deployed/slice membership; y:(n,) -> ordinal loss (n,)."""
    d = DIST[:, y].T                                   # (n,K): d[i,k]=|k-y_i|
    big = np.where(member, d, np.inf)
    mn = big.min(1)
    mn[~np.isfinite(mn)] = B                           # empty set -> B
    return mn

# --------------------------- threshold via event sweep ---------------------
def fit_threshold(S, y, cal_idx, group):
    """Exact t_j for group (list of labels) using cal rows with y in group.
       Returns (t_hat, fallback_bool, n_j)."""
    group = np.asarray(group)
    in_g = np.isin(y[cal_idx], group)
    rows = cal_idx[in_g]
    n_j = len(rows)
    target = ALPHA * (n_j + 1) - B                     # need running_sum <= target
    if n_j == 0 or target < 0:                         # infeasible even at loss 0
        return LAMBDA_MAX, True, n_j
    # events: including label k (k in group) at threshold s[i,k] sets point i's
    # loss to min(current, |k - y_i|). Sweep ascending score; first time the
    # aggregate loss sum <= target gives t_hat.
    cur = np.full(n_j, B)                              # empty slice
    ssum = cur.sum()
    if ssum <= target:                                 # already (cannot happen, B*n_j>target)
        return LAMBDA_MAX, False, n_j
    ev_s, ev_i, ev_d = [], [], []
    yy = y[rows]
    for col, k in enumerate(group):
        sc = S[rows, k]
        dd = np.abs(k - yy).astype(float)
        ev_s.append(sc); ev_i.append(np.arange(n_j)); ev_d.append(dd)
    ev_s = np.concatenate(ev_s); ev_i = np.concatenate(ev_i); ev_d = np.concatenate(ev_d)
    order = np.argsort(ev_s, kind="mergesort")
    ev_s, ev_i, ev_d = ev_s[order], ev_i[order], ev_d[order]
    m = len(ev_s); p = 0
    while p < m:
        s0 = ev_s[p]
        # process all events at this score
        while p < m and ev_s[p] == s0:
            i = ev_i[p]
            if ev_d[p] < cur[i]:
                ssum += ev_d[p] - cur[i]; cur[i] = ev_d[p]
            p += 1
        if ssum <= target:
            return float(s0), False, n_j
    return LAMBDA_MAX, False, n_j                      # feasible only at max slice

# --------------------------- deployment ------------------------------------
def deploy_union(S, groups, thr):
    """C_PW membership (N,K): k admitted iff s<=max_{j: k in G_j} t_j."""
    N = S.shape[0]
    Tmax = np.full(K, -np.inf)
    for g, t in zip(groups, thr):
        for k in g:
            if t > Tmax[k]:
                Tmax[k] = t
    return S <= Tmax[None, :]                          # (N,K) bool

def slice_membership(S, group, t):
    member = np.zeros_like(S, dtype=bool)
    for k in group:
        member[:, k] = S[:, k] <= t
    return member

# -------------------- observed loss-jump multiplicity -----------------------
def observed_max_jump_multiplicity(S, y, group):
    """Maximum distinct sample paths with a loss decrease at identical score.
    This audits the observed pool only; it is not a distributional theorem.
    """
    from collections import Counter
    c = Counter()
    group=np.asarray(group,dtype=int)
    for i in np.flatnonzero(np.isin(y,group)):
        order=np.argsort(S[i,group],kind="stable")
        vals=S[i,group][order]
        costs=np.abs(group[order]-y[i]).astype(float)
        current=B
        j=0
        while j<len(vals):
            t=vals[j]
            best=current
            while j<len(vals) and vals[j]==t:
                best=min(best,costs[j]);j+=1
            if best < current:
                c[float(t)] += 1
                current=best
    return max(c.values(),default=0)

# --------------------------- one arm on one split --------------------------
def run_arm(S, y, cal_idx, te_idx, groups):
    """Fit per-group thresholds on cal, deploy union, return per-group test
       risk/size/slice-risk/fallback for each group in `groups`."""
    thr, fb, nj = [], [], []
    for g in groups:
        t, f, n = fit_threshold(S, y, cal_idx, g)
        thr.append(t); fb.append(f); nj.append(n)
    dep = deploy_union(S, groups, thr)                 # (N,K)
    dep_te = dep[te_idx]
    dep_loss = losses_of_sets(dep_te, y[te_idx])
    dep_size = dep_te.sum(1)
    out = {}
    for gi, g in enumerate(groups):
        g = np.asarray(g)
        in_g = np.isin(y[te_idx], g)
        if in_g.sum() == 0:
            out[tuple(g)] = dict(risk=np.nan, size=np.nan, slice_risk=np.nan,
                                 delta=np.nan, fb=fb[gi], nj=nj[gi], n_te=0); continue
        sl = slice_membership(S[te_idx], g, thr[gi])
        sl_loss = losses_of_sets(sl, y[te_idx])
        R_dep = dep_loss[in_g].mean()
        R_slice = sl_loss[in_g].mean()
        group_slice_size=float(sl[in_g].sum(axis=1).mean())
        extra_size=float(dep_size[in_g].mean()-group_slice_size)
        if R_slice-R_dep > B*extra_size+1e-9:
            raise AssertionError("Risk-size inequality violated")
        out[tuple(g)] = dict(risk=R_dep, size=dep_size[in_g].mean(),
                             slice_risk=R_slice,delta=R_slice-R_dep,
                             slice_size=group_slice_size,size_inflation=extra_size,
                             fb=fb[gi],nj=nj[gi],n_te=int(in_g.sum()))
    # nesting check: slice subset of union => dep_loss <= slice_loss pointwise
    nest_ok = True
    for gi, g in enumerate(groups):
        sl = slice_membership(S[te_idx], np.asarray(g), thr[gi])
        if not np.all(losses_of_sets(dep_te, y[te_idx]) <= losses_of_sets(sl, y[te_idx]) + 1e-9):
            nest_ok = False
    return out, thr, nest_ok

# --------------------------- data ------------------------------------------
def load_or_simulate():
    if os.path.isfile(CSV_IN):
        recipe_path = os.environ.get("DDR_RECIPE", os.path.join(os.path.dirname(os.path.abspath(CSV_IN)), "ddr_scoring_recipe.json"))
        if not os.path.isfile(recipe_path):
            raise FileNotFoundError("No DDR scoring recipe next to the canonical score CSV; "
                                    "cannot certify input provenance.")
        with open(recipe_path, "r", encoding="utf-8") as f:
            recipe = json.load(f)
        with open(CSV_IN, "rb") as f:
            actual_sha256 = hashlib.sha256(f.read()).hexdigest()
        if recipe.get("score_csv_sha256") != actual_sha256:
            raise ValueError("Canonical score CSV SHA-256 does not match the scoring recipe")
        dataset_name = str(recipe.get("dataset", ""))
        if "DDR" not in dataset_name.upper():
            raise ValueError("Scoring recipe does not identify a DDR dataset")
        if any(t in dataset_name.upper() for t in ("MOCK", "SYNTHETIC", "SIMULATED", "TESTING")):
            raise ValueError("Scoring recipe identifies mock/test scores; refusing to label them real")
        df = pd.read_csv(CSV_IN)
        required = ["id", "grade"] + [f"p{k}" for k in range(K)]
        if any(k not in df.columns for k in required):
            raise ValueError(f"DDR score table must contain {required}; got {list(df.columns)}")
        if df[required].isna().any().any():
            raise ValueError("Missing values found in the DDR score table")
        if not df["id"].is_unique:
            raise ValueError("Repeated image ID in DDR score table")
        grades = pd.to_numeric(df["grade"], errors="raise").to_numpy(float)
        if not np.all(np.isfinite(grades) & (grades == grades.astype(int)) & (grades >= 0) & (grades < K)):
            raise ValueError("Invalid grade; expected integers 0,...,4")
        y = grades.astype(int)
        P = df[[f"p{k}" for k in range(K)]].to_numpy(float)
        if not np.all(np.isfinite(P)) or np.any(P < 0) or np.any(P > 1):
            raise ValueError("Nonfinite/out-of-range probability in canonical DDR scores")
        if not np.allclose(P.sum(axis=1), 1.0, atol=1e-5, rtol=0):
            raise ValueError("Probability rows do not sum to 1")
        P = P/P.sum(axis=1,keepdims=True)  # remove harmless floating-point error only
        recipe_pool_counts = recipe.get("counts",{}).get("pool")
        if recipe_pool_counts != np.bincount(y,minlength=K).tolist():
            raise ValueError("Canonical score-table grade counts disagree with scoring recipe")
        print(f"DDR INPUT WITH MATCHING SCORING RECIPE: {CSV_IN}: N={len(y)} counts={recipe_pool_counts}")
        return P, y, False, df["id"].astype(str).to_numpy()
    if os.environ.get("DDR_ALLOW_SIMULATED") != "1":
        raise FileNotFoundError(
            f"REAL DDR scores not found: {CSV_IN}. Refusing to simulate silently. "
            "For code testing ONLY, set DDR_ALLOW_SIMULATED=1; never use demo outputs in a manuscript."
        )
    # ---- EXPLICIT SIMULATION FOR UNIT TESTING ONLY ----
    rng = np.random.default_rng(0)
    counts = [1253, 126, 895, 47, 183]                 # real POOL per-grade counts
    y = np.concatenate([np.full(c, g) for g, c in enumerate(counts)])
    N = len(y)
    spread = 1.15                                      # tuned so QWK ~ 0.85
    logits = -spread * np.abs(np.arange(K)[None, :] - y[:, None]) \
             + rng.normal(0, 0.9, size=(N, K))
    P = np.exp(logits); P /= P.sum(1, keepdims=True)
    print(f"SYNTHETIC CODE TEST ONLY (no {CSV_IN}): N={N} counts={counts}")
    return P, y, True, np.asarray([f"SIMULATED_{i:06d}" for i in range(N)])

# ---------------- data-leakage prevention -----------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def apply_integrity_policy(P,y,ids,simulated):
    """Classify results as definitive only when duplicate handling was
    established before fitting; retaining an already-trained model and
    excluding contaminated pool IDs is sensitivity analysis, not retraining.
    """
    if simulated:
        return P,y,ids,"SYNTHETIC_DEMO_DO_NOT_PUBLISH",{"integrity_tier":"SYNTHETIC_TEST"}
    recipe_path = os.environ.get("DDR_RECIPE",os.path.join(os.path.dirname(os.path.abspath(CSV_IN)),"ddr_scoring_recipe.json"))
    with open(recipe_path,"r",encoding="utf-8") as stream:
        recipe = json.load(stream)
    dup = recipe.get("duplicate_audit",{})
    policy = dup.get("strategy")
    evidence={"input_score_sha256":sha256_file(CSV_IN),
              "recipe_sha256":sha256_file(recipe_path),
              "input_n":len(ids),
              "patient_independence":"UNVERIFIED_NO_PATIENT_IDS"}
    if policy == "content_deduplicate_before_split":
        folder=os.path.dirname(os.path.abspath(recipe_path))
        audit=os.path.join(folder,"ddr_content_audit.csv")
        manifest=os.path.join(folder,"ddr_split_assignment.csv")
        for path,key in ((audit,"content_audit_sha256"),(manifest,"split_manifest_sha256")):
            if not os.path.isfile(path) or sha256_file(path) != dup.get(key):
                raise ValueError(f"Missing or hash-mismatched duplicate-aware audit artifact: {path}")
        a=pd.read_csv(audit,dtype={"id":str,"representative_id":str,"sha256":str})
        m=pd.read_csv(manifest,dtype={"id":str})
        if not m["id"].is_unique or len(m)!=dup.get("retained_unique_images"):
            raise ValueError("Invalid unique-image split manifest")
        if set(m["split"])!={"train","val","pool"}:
            raise ValueError("Expected train,val,pool in unique-image manifest")
        if set(m.loc[m["split"]=="pool","id"]) != set(ids):
            raise ValueError("Score CSV IDs do not equal duplicate-aware pool manifest IDs")
        # No exact bytes across any retained partitions, and no duplicated
        # content among kept rows. Deleted duplicate IDs cannot enter fitting.
        retained=a[a["id"].isin(set(m["id"]))]
        if len(retained)!=len(m) or not retained["sha256"].is_unique:
            raise ValueError("Content duplication persists in retained model pool")
        status=dup.get("near_duplicate_review_status", "")
        if status == "CONSERVATIVE_PHASH_CANDIDATES_EXCLUDED":
            report_path=os.environ.get("DDR_PHASH_REPORT", "").strip()
            if not report_path or not os.path.isfile(report_path):
                raise ValueError("Automatic-screening run requires original DDR_PHASH_REPORT")
            if sha256_file(report_path)!=dup.get("phash_report_sha256"):
                raise ValueError("Conservative screening pHash report hash mismatch")
            r=pd.read_csv(report_path,dtype=str).fillna("")
            if not {"kind","ids"}.issubset(r.columns):
                raise ValueError("Invalid pHash candidate report")
            candidate_ids={v.strip() for x in r.loc[r.kind=="identical_phash","ids"]
                           for v in x.split(";") if v.strip()}
            excluded=a[a["split"]=="excluded_phash_candidate"]
            if set(excluded["id"])!=candidate_ids or (candidate_ids & set(m["id"])):
                raise ValueError("Missing/extraneous pHash candidate exclusions in scored split")
            if len(candidate_ids)!=dup.get("phash_candidate_ids_excluded"):
                raise ValueError("pHash candidate exclusion count mismatch")
            tier="REAL_DATA_DEDUP_RETRAINED_CONSERVATIVE"
            print(f"Conservative image-level screening verified: {len(candidate_ids)} pHash candidate images excluded")
        elif status == "PHASH_CANDIDATES_REVIEWED":
            tier="REAL_DATA_DEDUP_RETRAINED_REVIEWED"
        else:
            if os.environ.get("DDR_ALLOW_UNREVIEWED_NEARDUP") != "1":
                raise ValueError("Near-duplicate candidates not handled. Use reviewed pairs or automatic exclusion.")
            tier="PRELIMINARY_EXACT_DEDUP_NEARDUP_UNREVIEWED"
        evidence.update(integrity_tier="DEDUP_RETRAINED",audit_sha256=sha256_file(audit),
                        near_duplicate_review_status=status,
                        screening_policy=dup.get("screening_policy", "manual_review"))
        return P,y,ids,tier,evidence
    # Historical classifier: exact copies crossed fitting/pool. Never present
    # this model as a primary leakage-free fit, even after held-out exclusions.
    exclusions=os.environ.get("DDR_EXCLUSIONS","").strip()
    report=os.environ.get("DDR_DUPLICATE_REPORT","").strip()
    if not (exclusions and report):
        raise ValueError("The original DDR classifier used image-level splits with identified "
                         "cross-partition exact duplicates. Refusing publication-mode eval. "
                         "For a legacy exclusion SENSITIVITY ONLY supply DDR_EXCLUSIONS and DDR_DUPLICATE_REPORT; "
                         "for primary results retrain using make_ddr_scores_deduplicated.py.")
    excl=pd.read_csv(exclusions,dtype=str)
    if "id" not in excl.columns or excl["id"].isna().any() or not excl["id"].is_unique:
        raise ValueError("DDR_EXCLUSIONS must have unique nonmissing id column")
    excl_ids=set(excl["id"].tolist())
    if not excl_ids.issubset(set(ids)):
        raise ValueError("Exclusion list has IDs not in this original score pool")
    duplicate_report=pd.read_csv(report,dtype=str)
    required={"kind","ids","partitions"}
    if not required.issubset(duplicate_report.columns):
        raise ValueError(f"Duplicate report missing {required-set(duplicate_report.columns)}")
    implicated=set()
    # This report stores UNIQUE partitions for a cluster, not an image-level
    # partition column. Never zip 'ids' and 'partitions'. Identify all pool
    # members by intersecting the cluster's IDs with the scored pool itself.
    for row in duplicate_report.loc[duplicate_report["kind"]=="exact_bytes"].itertuples(index=False):
        cluster_ids={v.strip() for v in row.ids.split(";") if v.strip()}
        if len(cluster_ids)>1:
            implicated.update(cluster_ids & set(ids))
    needed=implicated
    if not needed.issubset(excl_ids):
        raise ValueError(f"Incomplete exact-image exclusion: {len(needed-excl_ids)} offending pool IDs remain")
    keep=np.array([i not in excl_ids for i in ids],dtype=bool)
    evidence.update(integrity_tier="LEGACY_EXACT_EXCLUSION_SENSITIVITY_ONLY",
                    exclusions_sha256=sha256_file(exclusions),
                    duplicate_report_sha256=sha256_file(report),
                    excluded_ids=len(excl_ids),required_exact_pool_ids=len(needed),
                    analyzed_n=int(keep.sum()))
    print(f"LEGACY exclusion sensitivity only: removed {len(excl_ids)} scored images, "
          f"N={keep.sum()}, NOT a verified leakage-free classifier")
    return P[keep],y[keep],ids[keep],"LEGACY_EXACT_EXCLUSIONS_SENSITIVITY_ONLY",evidence

# --------------------------- main ------------------------------------------
def main():
    P, y, simulated, ids = load_or_simulate()
    P, y, ids, mode, evidence = apply_integrity_policy(P, y, ids, simulated)
    # Separate output tiers forbid publication-oriented programs from picking up
    # legacy-exclusion or synthetic outputs by accident.
    output_dir = os.path.join(OUTDIR, mode,
                              "u_redrawn_each_split" if U_MODE == "redraw" else "u_fixed_once")
    if os.path.exists(os.path.join(output_dir, "ddr_crc_checks.txt")) and os.environ.get("DDR_OVERWRITE_RESULTS") != "1":
        raise FileExistsError("Evaluation output already exists; use a fresh DDR_CRC_OUT or set DDR_OVERWRITE_RESULTS=1")
    os.makedirs(output_dir, exist_ok=True)
    if not simulated:
        canonical_df = pd.read_csv(CSV_IN, dtype={"id":str})
        used = set(ids.tolist())
        analysis_df = canonical_df[canonical_df["id"].isin(used)].copy()
        analysis_df.to_csv(os.path.join(output_dir,"ddr_analysis_scores.csv"),index=False)
        evidence["analysis_scores_sha256"] = sha256_file(os.path.join(output_dir,"ddr_analysis_scores.csv"))
    with open(os.path.join(output_dir,"ddr_integrity_provenance.json"),"w") as f:
        json.dump(evidence,f,indent=2,sort_keys=True)
    N = len(y)
    sorted_idx = np.argsort(ids.astype(str), kind="stable")
    # The control condition uses exactly the old ID-stable fixed-U seed.
    if U_MODE == "fixed":
        U_fixed = np.empty(N, dtype=float)
        U_fixed[sorted_idx] = np.random.default_rng(SEED).random(N)
    else:
        U_fixed = None

    def scores_for_redraw(r):
        if U_MODE == "fixed":
            u = U_fixed
        else:
            # Independent seeded stream for each redraw, separate from splits.
            u = np.empty(N, dtype=float)
            u[sorted_idx] = np.random.default_rng(
                np.random.SeedSequence([SEED, 0x55AA, int(r)])
            ).random(N)
        sc = aps_scores(P, u)
        if not np.all(np.isfinite(sc)) or not np.all((sc >= -1e-10) & (sc <= LAMBDA_MAX)):
            raise ValueError("APS scores outside valid threshold range")
        return u, sc

    # reporting groups
    EXP_A = [[0], [1, 2], [3, 4]]                      # disjoint severity
    EXP_B = [[0], [1, 2, 3], [2, 3, 4], [3, 4]]        # overlapping clinical
    GRADES = [[k] for k in range(K)]                   # exact-grade refinement
    WHOLE  = [[0, 1, 2, 3, 4]]                          # marginal

    arms = {
        "marginal":        WHOLE,
        "exact_grade":     GRADES,     # = disjoint-refinement Mondrian (structured loss)
        "disjoint_sev":    EXP_A,      # Experiment A
        "overlap_union":   EXP_B,      # Experiment B (ours)
    }
    # groups we REPORT risk on (union of A and B strata + grades)
    report_groups = [(0,), (1,2), (3,4), (1,2,3), (2,3,4), (3,), (4,)]

    acc = {a: {g: {k: [] for k in ("risk","size","slice_risk","delta","fb")}
               for g in report_groups} for a in arms}
    nest_fail = 0
    raw_rows = []
    threshold_rows = []
    split_rows = []
    ux_rows = []
    max_M = {a: {tuple(gg): 0 for gg in gg_list} for a, gg_list in arms.items()}

    idx = np.arange(N)
    for r in range(N_REDRAW):
        rr = np.random.default_rng(SEED + 1 + r)
        perm = rr.permutation(idx)
        half = N // 2
        cal_idx, te_idx = perm[:half], perm[half:]
        # U is independent of the split and common to all evaluated arms here.
        U, S = scores_for_redraw(r)
        ux_rows.extend(dict(rep=r+1, id=str(ids[i]), grade=int(y[i]), u_x=float(U[i]))
                       for i in range(N))
        if r == 0 and U_MODE == "fixed":
            assert np.array_equal(U, U_fixed)
        assert len(cal_idx)==half and len(te_idx)==N-half and len(np.intersect1d(cal_idx,te_idx))==0
        split_rows.extend([dict(rep=r+1,id=str(ids[i]),fold="cal") for i in cal_idx])
        split_rows.extend([dict(rep=r+1,id=str(ids[i]),fold="test") for i in te_idx])
        # Maximum multiplicity across all score randomizations and group definitions.
        m_by_group = {tuple(gg): observed_max_jump_multiplicity(S, y, gg)
                      for gg in ([0,1,2,3,4], [0], [1,2], [3,4], [1,2,3], [2,3,4], [1], [2], [3], [4])}
        for a, groups in arms.items():
            for gg in groups:
                key = tuple(gg)
                max_M[a][key] = max(max_M[a][key], m_by_group[key])
            out, thr, nest_ok = run_arm(S, y, cal_idx, te_idx, groups)
            if not nest_ok:
                nest_fail += 1
                raise AssertionError(f"Pointwise slice domination failed: arm={a}, redraw={r+1}")
            for gi, gg in enumerate(groups):
                threshold_rows.append(dict(rep=r+1,arm=a,group="{"+",".join(map(str,gg))+"}",
                                           threshold=thr[gi],
                                           fallback=bool(out[tuple(gg)]["fb"]),
                                           n_cal=int(out[tuple(gg)]["nj"])))
            # map arm's own groups onto report groups by matching label sets
            for g in report_groups:
                gl = list(g)
                # risk of this arm on stratum g: deploy arm's union, condition y in g
                dep = deploy_union(S, groups, thr)[te_idx]
                in_g = np.isin(y[te_idx], np.asarray(gl))
                if in_g.sum() == 0:
                    continue
                dl = losses_of_sets(dep, y[te_idx])
                ds = dep.sum(1)
                r_value = float(dl[in_g].mean())
                sz_value = float(ds[in_g].mean())
                ncal = int(np.isin(y[cal_idx],np.asarray(gl)).sum())
                ntest = int(in_g.sum())
                acc[a][g]["risk"].append(r_value)
                acc[a][g]["size"].append(sz_value)
                raw = dict(rep=r+1,u_protocol=U_MODE,arm=a,group="{"+",".join(map(str,g))+"}",
                           risk=r_value,size=sz_value,n_cal=ncal,n_test=ntest,
                           slice_risk=np.nan,delta=np.nan,delta_size=np.nan,
                           rank_pred=np.nan,fallback=np.nan)
                # slice/delta/fb only meaningful if g is one of the arm's own groups
                if gl in [list(x) for x in groups]:
                    gi = [list(x) for x in groups].index(gl)
                    sl = slice_membership(S[te_idx], np.asarray(gl), thr[gi])
                    sr = losses_of_sets(sl, y[te_idx])[in_g].mean()
                    acc[a][g]["slice_risk"].append(sr)
                    acc[a][g]["delta"].append(sr-r_value)
                    fb_val = float(out[tuple(gl)]["fb"])
                    acc[a][g]["fb"].append(fb_val)
                    size_slice=float(slice_membership(S[te_idx],np.asarray(gl),thr[gi])[in_g].sum(axis=1).mean())
                    size_inflation=sz_value-size_slice
                    if sr-r_value>B*size_inflation+1e-9:
                        raise AssertionError("Risk-size bound failed on one redraw")
                    nlocal=out[tuple(gl)]["nj"]
                    rank_pred=(B*np.floor((nlocal+1)*ALPHA/B)/(nlocal+1)) if len(gl)==1 else np.nan
                    raw.update(slice_risk=sr,delta=sr-r_value,
                               delta_size=size_inflation,rank_pred=rank_pred,fallback=fb_val)
                raw_rows.append(raw)

    # ---- aggregate ----
    def ms(v):
        v = [x for x in v if np.isfinite(x)]
        return (np.mean(v), np.std(v,ddof=1)/np.sqrt(len(v)) if len(v)>1 else np.nan) if v else (np.nan, np.nan)
    rows = []
    for a in arms:
        for g in report_groups:
            rk = ms(acc[a][g]["risk"]); sz = ms(acc[a][g]["size"])
            sr = ms(acc[a][g]["slice_risk"]); dl = ms(acc[a][g]["delta"])
            fb = np.mean(acc[a][g]["fb"]) if acc[a][g]["fb"] else np.nan
            group_string="{"+",".join(map(str,g))+"}"
            group_raw=[z for z in raw_rows if z["arm"]==a and z["group"]==group_string]
            rows.append(dict(source="SIMULATED_DEMO" if simulated else ("DEDUP_RETRAINED" if evidence.get("integrity_tier")=="DEDUP_RETRAINED" else "LEGACY_EXACT_EXCLUSIONS_SENSITIVITY"),u_protocol=U_MODE,arm=a,group=group_string,
                             n_cal=np.mean([z["n_cal"] for z in group_raw]),
                             n_test=np.mean([z["n_test"] for z in group_raw]),
                             risk=rk[0],risk_se=rk[1],size=sz[0],size_se=sz[1],
                             slice_risk=sr[0],slice_risk_se=sr[1],delta=dl[0],delta_se=dl[1],
                             size_inflation=np.nanmean([z["delta_size"] for z in group_raw]) if np.any(np.isfinite([z["delta_size"] for z in group_raw])) else np.nan,
                             rank_pred=np.nanmean([z["rank_pred"] for z in group_raw]) if np.any(np.isfinite([z["rank_pred"] for z in group_raw])) else np.nan,
                             fallback_freq=fb))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(output_dir, "ddr_crc_summary.csv"), index=False)
    pd.DataFrame(raw_rows).to_csv(os.path.join(output_dir, "ddr_crc_redraws.csv"),index=False)
    pd.DataFrame(threshold_rows).to_csv(os.path.join(output_dir, "ddr_crc_thresholds.csv"),index=False)
    pd.DataFrame(split_rows).to_csv(os.path.join(output_dir, "ddr_crc_split_assignments.csv"),index=False)
    # Full U manifest: 100 x N rows for the primary design; one row per ID and redraw.
    pd.DataFrame(ux_rows).to_csv(os.path.join(output_dir,"ddr_crc_ux_by_redraw.csv"),index=False)
    # Pooled risk is measured on all test rows; compute from the disjoint
    # {0}, {1,2}, {3,4} partition so no groupwise averaging bias is introduced.
    marginal_raw = [z for z in raw_rows if z["arm"] == "marginal"]
    pooled_rows = []
    for rr in range(1, N_REDRAW+1):
        parts = [z for z in marginal_raw if z["rep"] == rr and z["group"] in ("{0}","{1,2}","{3,4}")]
        if len(parts) != 3 or sum(z["n_test"] for z in parts) != N-half:
            raise ValueError("Cannot reconstruct pooled marginal risk from the group partition")
        pooled_rows.append(dict(rep=rr, u_protocol=U_MODE,
                               pooled_marginal_risk=sum(z["risk"]*z["n_test"] for z in parts)/(N-half),
                               pooled_marginal_size=sum(z["size"]*z["n_test"] for z in parts)/(N-half)))
    pd.DataFrame(pooled_rows).to_csv(os.path.join(output_dir, "ddr_crc_pooled_by_redraw.csv"),index=False)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("\n=== per-arm per-stratum (mean over {} redraws, alpha={}) ===".format(N_REDRAW, ALPHA))
    print(summ.to_string(index=False,
          float_format=lambda x: f"{x:.4f}" if np.isfinite(x) else "  . "))

    # ---- checks ----
    with open(os.path.join(output_dir, "ddr_crc_checks.txt"), "w") as f:
        f.write(f"U protocol={U_MODE}; U seed base={SEED}; "
                f"redraws={N_REDRAW}; shared within redraw across candidate labels and arms\n")
        f.write("Split stream: numpy.default_rng(SEED + 1 + r), with r zero-indexed; "
                "identical across both U protocols\n")
        f.write(f"source={'SIMULATED_DEMO_DO_NOT_PUBLISH' if simulated else evidence['integrity_tier']} N={N} alpha={ALPHA} B={B} redraws={N_REDRAW}\n")
        if not simulated:
            with open(CSV_IN,'rb') as inp: f.write(f"score_csv_sha256={hashlib.sha256(inp.read()).hexdigest()}\n")
        f.write(f"data_integrity_tier={mode}\n")
        f.write(f"analysis_pool_n={N}; analyzed_pool_sha256={evidence.get('analysis_scores_sha256','SYNTHETIC')}\n")
        f.write(f"python={platform.python_version()} numpy={np.__version__} pandas={pd.__version__}\n")
        f.write(f"pointwise nesting (dep_loss<=slice_loss) failures: {nest_fail}\n")
        f.write("Maximum observed simultaneous loss-jump multiplicity across all U redraws:\n")
        for arm, grps in arms.items():
            for gg in grps:
                f.write(f"  {arm} {gg}: M_observed_max={max_M[arm][tuple(gg)]}\n")
        f.write("Risk-size inequality checked on all evaluated group/redraw pairs.\n")
        pooled_values = np.array([z["pooled_marginal_risk"] for z in pooled_rows])
        f.write(f"Pooled marginal ordinal risk: {pooled_values.mean():.6f} "
                f"(MC SE={pooled_values.std(ddof=1)/np.sqrt(N_REDRAW):.6f})\n")
        f.write(f"B/alpha support barrier = {B/ALPHA:.1f} (need n_j+1 >= this to certify)\n")
        mg = summ[summ.arm=="marginal"].set_index("group")["risk"]
        f.write(f"\nmarginal risk on severe {{3,4}}: {mg.get('{3,4}', np.nan):.4f}  (alpha={ALPHA})\n")
        og = summ[summ.arm=="overlap_union"].set_index("group")
        for g in ("{1,2,3}", "{2,3,4}", "{3,4}", "{0}"):
            if g in og.index:
                f.write(f"overlap {g}: risk={og.loc[g,'risk']:.4f}  "
                        f"slice={og.loc[g,'slice_risk']:.4f}  delta={og.loc[g,'delta']:.4f}  "
                        f"fb={og.loc[g,'fallback_freq']}\n")
        eg = summ[summ.arm=="exact_grade"].set_index("group")
        if "{3}" in eg.index:
            f.write(f"\nexact-grade stratum {{3}}: risk={eg.loc['{3}','risk']:.4f}  "
                    f"size={eg.loc['{3}','size']:.4f}  fallback_freq={eg.loc['{3}','fallback_freq']}\n")
    print(f"\nwrote detailed evaluation artifacts to {output_dir}")
    if mode not in ("REAL_DATA_DEDUP_RETRAINED_REVIEWED", "REAL_DATA_DEDUP_RETRAINED_CONSERVATIVE"):
        print("IMPORTANT: These outputs are NOT certified as a leakage-free definitive patient-level analysis.")
    print(f"pointwise-nesting failures across all arms/redraws: {nest_fail}")

if __name__ == "__main__":
    main()
