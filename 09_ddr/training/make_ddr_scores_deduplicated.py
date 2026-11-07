#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================================
# make_ddr_scores.py   (Windows-safe: all execution under __main__ guard)
# ---------------------------------------------------------------------------
# Produce the CANONICAL probability table for the DDR diabetic-retinopathy
# grading experiment used by the group-conditional CRC Section 5 code.
#
# Data in hand:
#   IMG_DIR   flat folder of ~12,524 JPGs (both DDR naming schemes coexist)
#   CSV_PATH  DR_grading.csv with columns id_code,diagnosis (grades 0..4;
#             ungradable already removed; 12,522 labelled rows)
#
# Exact-byte duplicate image content is discarded BEFORE our deterministic, grade-
# stratified split:
#     TRAIN (70%)  -> fit the classifier
#     VAL   (10%)  -> model selection (quadratic-weighted kappa)
#     POOL  (20%)  -> held-out SCORED POOL; the ONLY rows written out and the
#                     ONLY rows the CRC code ever sees (no leakage). All CRC
#                     calibration/test redraws later are UNIFORM within POOL.
#
# Output (durable artifact; treat as data, never regenerate to reproduce):
#   scores_ddr_pool.csv       id, grade, p0, p1, p2, p3, p4
#   ddr_scoring_recipe.json   seeds, arch, split fractions, counts, metrics
#   ddr_model_best.pt         checkpoint (provenance only)
#   ddr_split_assignment.csv  id -> {train,val,pool}
#
# Reproducibility: fixed seeds + cudnn deterministic flags + fp32 inference (no AMP)
#   reduce run-to-run variation on one machine. Bitwise reproducibility is NOT
#   guaranteed across different GPUs, drivers, or library versions (torch is called
#   with warn_only=True below, so nondeterministic kernels warn rather than error).
# Dependencies (do NOT reinstall a working CUDA torch):
#   pip install torchvision pandas pillow numpy scikit-learn tqdm
# ===========================================================================

import os, json, random, time, hashlib, platform
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, confusion_matrix
from tqdm import tqdm

# --------------------------- CONFIG ----------------------------------------
# Set these via environment variables to point at your local DDR data release.
#   export DDR_IMG_DIR=/path/to/DDR/DR_grading   DDR_CSV=/path/to/DDR/DR_grading.csv
IMG_DIR  = os.environ.get("DDR_IMG_DIR", "PATH/TO/DDR/DR_grading")
CSV_PATH = os.environ.get("DDR_CSV",     "PATH/TO/DDR/DR_grading.csv")
if "PATH/TO/DDR" in IMG_DIR or "PATH/TO/DDR" in CSV_PATH:
    raise SystemExit("Set DDR_IMG_DIR and DDR_CSV to your local DDR image folder and label CSV.")

OUTDIR     = os.environ.get("DDR_OUTDIR", "./ddr_scores_out")
POOL_FRAC  = 0.20
VAL_FRAC   = 0.10
IMG_SIZE   = 224
BATCH      = 32
EPOCHS     = 25
LR         = 3e-4
WD         = 1e-4
NUM_WORKERS= 8              # set to 0 if Windows workers still misbehave
SEED       = 20260801
NCLASS     = 5
USE_AMP    = False
# ---------------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def worker_init_fn(wid):
    # Use DataLoader's per-epoch/per-worker seed; do not reset each worker to
    # the same augmentation random stream at every epoch.
    s = torch.initial_seed() % (2**32)
    np.random.seed(s); random.seed(s)

def counts(df):
    return df["grade"].value_counts().reindex(range(NCLASS), fill_value=0).tolist()

def resolve(fn):
    p = os.path.join(IMG_DIR, fn)
    if os.path.isfile(p):
        return p
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        if os.path.isfile(p + ext):
            return p + ext
    return None

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF_TRAIN = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(), NORM,
])
TF_EVAL = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(), NORM,
])

class DDR(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True); self.tfm = tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tfm(Image.open(r["path"]).convert("RGB")), int(r["grade"]), r["id"]

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval(); ys, ps = [], []
    for x, y, _ in dl:
        x = x.to(device, non_blocking=True)
        ps.append(torch.softmax(model(x).float(), 1).cpu().numpy()); ys.append(y.numpy())
    P = np.concatenate(ps); Y = np.concatenate(ys); yhat = P.argmax(1)
    return cohen_kappa_score(Y, yhat, weights="quadratic"), balanced_accuracy_score(Y, yhat), P, Y


# --- Content-based de-duplication performed BEFORE any data split. ---
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
    def root(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def join(self, x, y):
        a, b = self.root(x), self.root(y)
        if a != b: self.parent[max(a,b)] = min(a,b)

def dedup_before_split(lab, reviewed_pairs_path=None):
    """Retain one representative per exact-byte/confirmed-same-image cluster.
    Reviewed pairs CSV has id_a,id_b,is_duplicate (yes/no).
    Hash collisions are cryptographically negligible. Distinct patients/eyes with
    near-identical images cannot be identified from filename and SHA alone.
    """
    hashes = [sha256_file(p) for p in lab["path"]]
    uf = DSU(len(lab))
    first = {}
    for i, h in enumerate(hashes):
        if h in first: uf.join(first[h], i)
        else: first[h] = i
    reviewed_n = 0
    if reviewed_pairs_path:
        pairs = pd.read_csv(reviewed_pairs_path, dtype=str).fillna("")
        if not {"id_a", "id_b", "is_duplicate"}.issubset(pairs.columns):
            raise ValueError("Reviewed pairs needs id_a,id_b,is_duplicate columns")
        by_id = {str(k): i for i, k in enumerate(lab["id"])}
        for row in pairs.itertuples(index=False):
            a, b = str(row.id_a), str(row.id_b)
            if a not in by_id or b not in by_id:
                raise ValueError(f"Reviewed pair image not in label table: {a}, {b}")
            decision = str(row.is_duplicate).strip().lower()
            if decision not in ("yes", "no"):
                raise ValueError("All reviewed decisions must be yes or no")
            if decision == "yes": uf.join(by_id[a], by_id[b])
            reviewed_n += 1
    clusters = defaultdict(list)
    for i in range(len(lab)): clusters[uf.root(i)].append(i)
    retained, excluded, cluster_ids = [], [], {}
    for members in clusters.values():
        grades = {int(lab.at[i, "grade"]) for i in members}
        if len(grades) != 1:
            names=[lab.at[i, "id"] for i in members]
            raise ValueError(f"Conflicting clinical labels in same-image cluster: {names} / grades={grades}. Manually adjudicate.")
        representative = min(members, key=lambda i: str(lab.at[i,"id"]))
        retained.append(representative)
        for i in members:
            cluster_ids[i]=str(lab.at[representative,"id"])
            if i != representative: excluded.append(i)
    audit = lab[["id","grade"]].copy()
    audit["sha256"] = hashes
    audit["representative_id"] = [cluster_ids[i] for i in range(len(lab))]
    audit["deduplicated_out"] = [i in set(excluded) for i in range(len(lab))]
    keep = lab.iloc[sorted(retained)].copy().reset_index(drop=True)
    meta = {"strategy":"content_deduplicate_before_split",
            "full_original_rows":len(lab),"retained_unique_images":len(keep),
            "excluded_duplicate_rows":len(excluded),"cluster_count":len(clusters),
            "reviewed_pairs_rows":reviewed_n,
            "reviewed_pairs_sha256":sha256_file(reviewed_pairs_path) if reviewed_pairs_path else None,
            "near_duplicate_review_status": "REVIEWED_PAIRS_PROVIDED" if reviewed_pairs_path else "NOT_REVIEWED",
            "patient_overlap_status":"UNKNOWN_NO_PATIENT_IDENTIFIERS"}
    return keep,audit,meta

def exclude_all_phash_candidates(lab, report_path):
    """Conservative screening: exclude EVERY image in an identical-pHash
    candidate cluster, including nonduplicates. This is an objective,
    reproducible rule; it is NOT visual adjudication or patient-level dedup.
    """
    if not report_path or not os.path.isfile(report_path):
        raise FileNotFoundError("Automatic candidate exclusion requires the original ddr_duplicate_report.csv")
    report = pd.read_csv(report_path, dtype=str).fillna("")
    if not {"kind", "ids"}.issubset(report.columns):
        raise ValueError("Duplicate report requires kind and ids columns")
    suspects = report.loc[report["kind"] == "identical_phash", "ids"]
    if len(suspects) == 0:
        raise ValueError("Report has no identical_phash groups; refusing silent zero-exclusion run")
    candidate_ids = {v.strip() for group in suspects for v in group.split(";") if v.strip()}
    known_ids = set(lab["id"].astype(str))
    missing = candidate_ids - known_ids
    if missing:
        raise ValueError(f"{len(missing)} pHash candidate IDs missing from DDR labels; wrong report/dataset version. Examples: {sorted(missing)[:5]}")
    mask = lab["id"].astype(str).isin(candidate_ids)
    eliminated, keep = lab.loc[mask].copy(), lab.loc[~mask].copy().reset_index(drop=True)
    if not len(eliminated) or not len(keep):
        raise ValueError("Invalid automatic candidate exclusions")
    # Archive precisely which observations were discarded, including their grades.
    return keep, eliminated, {"status": "CONSERVATIVE_PHASH_CANDIDATES_EXCLUDED",
        "excluded_candidate_images": int(len(eliminated)),
        "candidate_groups": int(len(suspects)),
        "excluded_ids_sha256": hashlib.sha256("\n".join(sorted(candidate_ids)).encode()).hexdigest(),
        "phash_report_sha256": sha256_file(report_path)}

def grade_group_split(lab, seed):
    """Approximate 70/10/20 grade-stratified split of UNIQUE images.
    All exact/confirmed duplicate content is already reduced to one row.
    """
    rng = np.random.default_rng(seed)
    split = np.array(["train"]*len(lab),dtype=object)
    for gr in range(NCLASS):
        ids=np.flatnonzero(lab["grade"].to_numpy()==gr)
        rng.shuffle(ids)
        n_pool=round(POOL_FRAC*len(ids)); n_val=round(VAL_FRAC*len(ids))
        split[ids[:n_pool]]="pool"
        split[ids[n_pool:n_pool+n_val]]="val"
    return split

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if os.path.exists(os.path.join(OUTDIR, "scores_ddr_pool.csv")) and os.environ.get("DDR_OVERWRITE") != "1":
        raise FileExistsError("Canonical scores_ddr_pool.csv already exists; refusing to overwrite. "
                              "Set DDR_OVERWRITE=1 only after explicitly archiving the existing run.")
    seed_everything(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: no CUDA device found; this is meant to run on your GPU.")

    # ---- labels + image paths ----
    lab = pd.read_csv(CSV_PATH, dtype={"id_code":str})
    assert {"id_code", "diagnosis"}.issubset(lab.columns), \
        f"CSV must have id_code,diagnosis; got {list(lab.columns)}"
    lab = lab.rename(columns={"id_code": "id", "diagnosis": "grade"})
    if lab["id"].isna().any() or not lab["id"].is_unique:
        raise ValueError("Missing or duplicate image IDs in label table")
    numeric_grade=pd.to_numeric(lab["grade"],errors="raise")
    if numeric_grade.isna().any() or not numeric_grade.isin(range(5)).all():
        raise ValueError("Grades must be exact integers 0,...,4")
    lab["grade"] = numeric_grade.astype(int)
    if len(lab) != 12522:
        raise ValueError(f"Expected 12522 graded DDR rows; found {len(lab)}. Check dataset version.")
    lab["path"] = lab["id"].map(resolve)
    miss = lab["path"].isna().sum()
    if miss:
        raise FileNotFoundError(f"{miss} labelled images not found under {IMG_DIR}")
    if not lab["path"].is_unique:
        raise ValueError("Different label rows resolve to the same image file; possible leakage")
    # Distinct filenames alone do not exclude duplicated image content or
    # multiple images from one patient; those require a separate data audit.
    print(f"labelled images joined to files: {len(lab)}")
    print("full per-grade counts (0..4):", counts(lab), " total", len(lab))

    # The source may contain identical files under DIFFERENT IDs. We must
    # de-duplicate by CONTENT before allocating ANY training/validation/pool rows.
    original_count = counts(lab)
    original_rows = len(lab)
    near_review = os.environ.get("DDR_REVIEWED_DUPLICATES", "").strip() or None
    automatic = os.environ.get("DDR_AUTOMATIC_NEARDUP_EXCLUSION") == "1"
    if automatic and near_review:
        raise ValueError("Choose automatic candidate exclusion OR manual review, not both")
    if automatic:
        lab, phash_excluded, auto_meta = exclude_all_phash_candidates(
            lab, os.environ.get("DDR_PHASH_REPORT", "").strip())
        print(f"AUTOMATIC: excluded {len(phash_excluded)} near-duplicate candidate images "+
              f"from {auto_meta['candidate_groups']} pHash clusters BEFORE splitting")
    else:
        phash_excluded = None
        auto_meta = None
        if near_review and not os.path.isfile(near_review):
            raise FileNotFoundError(near_review)
        if os.environ.get("DDR_NEARDUP_REVIEW_COMPLETE") == "1" and not near_review:
            raise ValueError("Manual review signoff requires a reviewed CSV")
    lab, dedup_audit, dup_meta = dedup_before_split(lab, near_review)
    if automatic:
        dup_meta.update({
            "strategy": "content_deduplicate_before_split",
            "full_original_rows": original_rows,
            "excluded_near_duplicate_rows": len(phash_excluded),
            "near_duplicate_review_status": "CONSERVATIVE_PHASH_CANDIDATES_EXCLUDED",
            "phash_report_sha256": auto_meta["phash_report_sha256"],
            "phash_candidate_groups": auto_meta["candidate_groups"],
            "phash_candidate_ids_excluded": auto_meta["excluded_candidate_images"],
            "phash_excluded_ids_sha256": auto_meta["excluded_ids_sha256"],
            "screening_policy": "EXCLUDE_ALL_UNREVIEWED_PHASH_CANDIDATE_IDS_BEFORE_SPLIT",
            "near_duplicate_review_performed": False,
        })
    elif near_review and os.environ.get("DDR_NEARDUP_REVIEW_COMPLETE") == "1":
        dup_meta["near_duplicate_review_status"] = "PHASH_CANDIDATES_REVIEWED"
        phash_report = os.environ.get("DDR_PHASH_REPORT", "").strip()
        if not phash_report or not os.path.isfile(phash_report):
            raise ValueError("Manual review signoff requires DDR_PHASH_REPORT")
        from itertools import combinations
        needed=set()
        report=pd.read_csv(phash_report,dtype=str).fillna("")
        for ids_text in report.loc[report.kind=="identical_phash", "ids"]:
            vals=[v.strip() for v in ids_text.split(";") if v.strip()]
            needed.update(frozenset(pair) for pair in combinations(vals,2))
        reviewed=pd.read_csv(near_review,dtype=str).fillna("")
        seen={frozenset((a,b)) for a,b in zip(reviewed.id_a,reviewed.id_b)}
        if needed - seen:
            raise ValueError(f"Near-duplicate manual review incomplete: {len(needed-seen)} pairs")
        dup_meta["phash_report_sha256"] = sha256_file(phash_report)
        dup_meta["phash_candidate_pairs_reviewed"] = len(needed)
    lab = lab.reset_index(drop=True)
    lab["split"] = grade_group_split(lab, SEED)
    assignments = lab[["id","grade","split"]].copy()
    dedup_audit["split"] = dedup_audit["representative_id"].map(
        dict(zip(lab["id"],lab["split"])))
    dedup_audit.loc[dedup_audit["deduplicated_out"],"split"]="deduplicated_out"
    if automatic:
        removed = phash_excluded[["id", "grade"]].copy()
        removed["sha256"] = [sha256_file(path) for path in phash_excluded["path"]]
        removed["representative_id"] = removed["id"]
        removed["deduplicated_out"] = True
        removed["split"] = "excluded_phash_candidate"
        dedup_audit = pd.concat([dedup_audit, removed], ignore_index=True)
    # Discarded candidate images must never appear in model fitting or scoring.
    if automatic and (set(phash_excluded["id"]) & set(assignments["id"])):
        raise AssertionError("An excluded pHash candidate entered the split manifest")
    audit_path=os.path.join(OUTDIR,"ddr_content_audit.csv")
    dedup_audit.to_csv(audit_path,index=False)
    assignments.to_csv(os.path.join(OUTDIR, "ddr_split_assignment.csv"), index=False)
    # Verify that neither original exact hashes nor confirmed duplicate clusters
    # can cross any of the three disjoint model/evaluation partitions.
    split_by_rep=dict(zip(lab["id"],lab["split"]))
    audit_reps = dedup_audit["representative_id"].map(split_by_rep)
    if (dedup_audit.groupby("sha256").apply(
       lambda z: audit_reps.loc[z.index].nunique(),include_groups=False)>1).any():
        raise AssertionError("Identical file bytes crossed partitions")
    if not assignments["id"].is_unique:
        raise AssertionError("Non-unique content in retained split manifest")

    df_tr = lab[lab.split == "train"].reset_index(drop=True)
    df_va = lab[lab.split == "val"].reset_index(drop=True)
    df_po = lab[lab.split == "pool"].reset_index(drop=True)
    dup_meta["content_audit_sha256"]=sha256_file(audit_path)
    dup_meta["split_manifest_sha256"]=sha256_file(os.path.join(OUTDIR,"ddr_split_assignment.csv"))
    print(f"Content duplicates excluded BEFORE split: {dup_meta['excluded_duplicate_rows']}")
    print("split per-grade counts (0..4):")
    print("  train:", counts(df_tr), " n=", len(df_tr))
    print("  val  :", counts(df_va), " n=", len(df_va))
    print("  POOL :", counts(df_po), " n=", len(df_po))
    print("WARNING: exact-byte deduplication DOES NOT imply patient-level independence.")
    if not near_review and not automatic:
        print("WARNING: near-duplicate candidate review NOT supplied; tag results as preliminary.")

    gen = torch.Generator(); gen.manual_seed(SEED)
    dl_tr = DataLoader(DDR(df_tr, TF_TRAIN), batch_size=BATCH, shuffle=True,
                       num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn,
                       generator=gen, pin_memory=True)
    dl_va = DataLoader(DDR(df_va, TF_EVAL), batch_size=BATCH, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)
    dl_po = DataLoader(DDR(df_po, TF_EVAL), batch_size=BATCH, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)

    # ---- model ----
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NCLASS)
    model = model.to(device)

    cls_n = torch.tensor(counts(df_tr), dtype=torch.float)
    cls_w = (cls_n.sum() / (NCLASS * cls_n.clamp(min=1))).to(device)
    criterion = nn.CrossEntropyLoss(weight=cls_w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    # ---- train + select on VAL ----
    best = {"qwk": -1.0, "epoch": -1}; hist = []
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); run = 0.0
        for x, y, _ in tqdm(dl_tr, desc=f"epoch {ep+1}/{EPOCHS}", leave=False):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item() * x.size(0)
        sched.step()
        qwk, bacc, _, _ = evaluate(model, dl_va, device)
        hist.append({"epoch": ep+1, "train_loss": run/len(df_tr), "val_qwk": qwk,
                     "val_bacc": bacc, "sec": time.time()-t0})
        print(f"  ep{ep+1}: train_loss={run/len(df_tr):.4f}  val_qwk={qwk:.4f}  val_bacc={bacc:.4f}")
        if qwk > best["qwk"]:
            best = {"qwk": qwk, "bacc": bacc, "epoch": ep+1}
            torch.save(model.state_dict(), os.path.join(OUTDIR, "ddr_model_best.pt"))
    print(f"best val QWK={best['qwk']:.4f} at epoch {best['epoch']}")

    # ---- frozen inference on POOL ----
    model.load_state_dict(torch.load(os.path.join(OUTDIR, "ddr_model_best.pt"), map_location=device))
    po_qwk, po_bacc, P, Y = evaluate(model, dl_po, device)
    print(f"POOL (held-out scored pool): QWK={po_qwk:.4f}  bACC={po_bacc:.4f}")

    out = pd.DataFrame({"id": df_po["id"].values, "grade": Y.astype(int)})
    for k in range(NCLASS):
        out[f"p{k}"] = P[:, k]
    pc = [f"p{k}" for k in range(NCLASS)]
    out[pc] = out[pc].clip(lower=0)
    assert np.allclose(out[pc].sum(1), 1.0, atol=1e-4), "softmax rows do not sum to 1"
    scores_path=os.path.join(OUTDIR,"scores_ddr_pool.csv")
    out.to_csv(scores_path,index=False)

    def file_hash(path):
        h=hashlib.sha256()
        with open(path,'rb') as inp:
            for chunk in iter(lambda:inp.read(1024*1024),b''):
                h.update(chunk)
        return h.hexdigest()

    recipe = {
        "dataset": "OIA-DDR DR grading (flat repackaging: images + DR_grading.csv)",
        "input_label_csv_sha256":file_hash(CSV_PATH),
        "score_csv_sha256":file_hash(scores_path),
        "split_manifest_sha256":file_hash(os.path.join(OUTDIR,"ddr_split_assignment.csv")),
        "content_audit_sha256":file_hash(audit_path),
        "checkpoint_sha256":file_hash(os.path.join(OUTDIR,"ddr_model_best.pt")),
        "environment":{"python":platform.python_version(),"torch":torch.__version__,
                       "torchvision":torchvision.__version__,"numpy":np.__version__,
                       "pandas":pd.__version__,"device":device,"cuda":torch.version.cuda},
        "duplicate_audit":dup_meta,
        "caveat":"All pHash candidate images excluded before split; exact-content duplicates deduplicated; patient-level independence cannot be verified.",
        "arch": "efficientnet_b0 (ImageNet-pretrained)", "nclass": NCLASS, "img_size": IMG_SIZE,
        "split": {"train_frac": 1-POOL_FRAC-VAL_FRAC, "val_frac": VAL_FRAC, "pool_frac": POOL_FRAC,
                  "stratified_by_grade": True, "seed": SEED},
        "counts": {"full": original_count, "unique_after_content_dedup":counts(lab), "train": counts(df_tr), "val": counts(df_va), "pool": counts(df_po)},
        "batch": BATCH, "epochs": EPOCHS, "lr": LR, "weight_decay": WD, "use_amp": USE_AMP,
        "selection_metric": "val quadratic-weighted kappa",
        "class_weights_inverse_freq": cls_w.detach().cpu().tolist(),
        "best_val": best, "pool_qwk": po_qwk, "pool_bacc": po_bacc,
        "pool_confusion": confusion_matrix(Y, P.argmax(1)).tolist(), "history": hist,
    }
    with open(os.path.join(OUTDIR,"ddr_scoring_recipe.json"),"w") as f:
        json.dump(recipe,f,indent=2)
    print(f"\nwrote {os.path.join(OUTDIR,'scores_ddr_pool.csv')}  ({len(out)} rows)")
    print("wrote ddr_scoring_recipe.json, ddr_split_assignment.csv, ddr_content_audit.csv, ddr_model_best.pt")
    print("Upload scores_ddr_pool.csv back to the chat to run the CRC evaluation.")

if __name__ == "__main__":
    main()
