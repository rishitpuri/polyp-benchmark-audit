"""Leak-safe grouping, specialised to how each dataset is actually used.

A first attempt thresholded the thumbnail-similarity graph and took connected
components. That failed: thumbnail correlations are high enough that the graph
percolates, and by tau = 0.77 all 612 CVC-ClinicDB frames collapse into one
component. Completeness is then 1.0 trivially and the grouping is useless. The
contiguity-constrained DP of script 03 does not chain, so it is the right tool
wherever frames are ordered.

The three datasets play different roles, and each needs something different:

  CVC-ClinicDB  used for *training*, and the dataset whose random splits leak.
                Its official frame-range table exists, so no recovery is
                needed: group by the 29 official sequences.
  Kvasir-SEG    used for training under cross-validation. Filenames are
                content hashes, so no ordered method applies and no ground
                truth exists. Leak-freedom cannot be certified here; we emit
                groupings at several strictness levels and let the downstream
                experiment report sensitivity instead of a single number.
  ColonDB /     used only as zero-shot *test* sets. Folds are irrelevant;
  ETIS / CVC-300 what matters is removing duplicates and the CVC-300 overlap.

The conservative-K rule for ordered datasets is calibrated on CVC-ClinicDB,
exploiting the asymmetry that under-segmentation is safe (two sequences merged
both land on the same side) while over-segmentation is not.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import homogeneity_completeness_v_measure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "contamination"
CACHE = OUT / "features"
SPLITS = ROOT / "outputs" / "splits"

CLINICDB_SEQUENCES = [
    (1, 25), (26, 50), (51, 67), (68, 78), (79, 103), (104, 126), (127, 151),
    (152, 177), (178, 199), (200, 205), (206, 227), (228, 252), (253, 277),
    (278, 297), (298, 317), (318, 342), (343, 363), (364, 383), (384, 408),
    (409, 428), (429, 447), (448, 466), (467, 478), (479, 503), (504, 528),
    (529, 546), (547, 571), (572, 591), (592, 612),
]
K_RATIOS = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.041, 0.047, 0.055, 0.065]
KVASIR_TAUS = [0.90, 0.93, 0.95, 0.97, 0.98, 0.99]
N_FOLDS = 5
GIANT_FRAC = 0.25  # a component holding >25% of images means the graph percolated


def segment_costs(X):
    n = len(X)
    s1 = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, axis=0)])
    s2 = np.concatenate([[0.0], np.cumsum((X ** 2).sum(1))])
    cost = np.full((n + 1, n + 1), np.inf)
    for i in range(n):
        j = np.arange(i + 1, n + 1)
        block = s1[j] - s1[i]
        cost[i, j] = (s2[j] - s2[i]) - (block ** 2).sum(1) / (j - i)
    return np.maximum(cost, 0.0)


def dp_labels(cost, n, k):
    best = np.full((k + 1, n + 1), np.inf)
    back = np.zeros((k + 1, n + 1), dtype=int)
    best[0, 0] = 0.0
    for kk in range(1, k + 1):
        for j in range(kk, n + 1):
            cand = best[kk - 1, :j] + cost[:j, j]
            i = int(np.argmin(cand))
            best[kk, j], back[kk, j] = cand[i], i
    bounds, j = [], n
    for kk in range(k, 0, -1):
        i = back[kk, j]
        bounds.append((i, j))
        j = i
    labels = np.zeros(n, dtype=int)
    for seq_id, (i, j) in enumerate(reversed(bounds), start=1):
        labels[i:j] = seq_id
    return labels


def clinicdb_truth(frames):
    lab = np.zeros(len(frames), dtype=int)
    for i, (lo, hi) in enumerate(CLINICDB_SEQUENCES, start=1):
        lab[(frames >= lo) & (frames <= hi)] = i
    return lab


def make_folds(groups, n_folds, seed=0):
    sizes = pd.Series(groups).value_counts()
    order = sizes.index.values[np.argsort(-sizes.values, kind="stable")]
    loads = np.zeros(n_folds, dtype=int)
    assign = {}
    for g in order:
        f = int(np.argmin(loads))
        assign[g] = f
        loads[f] += int(sizes[g])
    return np.array([assign[g] for g in groups]), loads


def load_feat(name):
    d = np.load(CACHE / f"{name}.npz", allow_pickle=True)
    return list(d["images"]), d["X"]


def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(OUT / "image_index.csv")
    corr = np.load(OUT / "thumb_corr.npy").astype(np.float64)
    manifest = {}

    # ---------- 1. calibrate the conservative-K rule on CVC-ClinicDB ----------
    names_c, Xc = load_feat("cvc-clinicdb")
    frames_c = np.array([int(re.sub(r"\D", "", s)) for s in names_c])
    truth_c = clinicdb_truth(frames_c)
    nc = len(Xc)
    cost_c = segment_costs(Xc)

    print(f"=== conservative-K calibration on CVC-ClinicDB (n={nc}, true K=29, ratio={29/nc:.4f}) ===")
    rows = []
    for ratio in K_RATIOS:
        k = max(2, int(round(ratio * nc)))
        lab = dp_labels(cost_c, nc, k)
        hom, comp, _ = homogeneity_completeness_v_measure(truth_c, lab)
        rows.append({"K_ratio": ratio, "K": k, "homogeneity": round(hom, 4),
                     "completeness": round(comp, 4), "leak_safe": comp >= 0.99999})
    cal = pd.DataFrame(rows)
    print(cal.to_string(index=False))
    cal.to_csv(OUT / "kratio_calibration_clinicdb.csv", index=False)

    safe = cal[cal["leak_safe"]]
    if safe.empty:
        print("\n  WARNING: no ratio achieved completeness 1.0")
        ratio_star = float(cal.loc[cal["completeness"].idxmax(), "K_ratio"])
    else:
        ratio_star = float(safe["K_ratio"].max())
        margin = (29 / nc) / ratio_star
        print(f"\n  completeness 1.0 up to K_ratio = {ratio_star:.3f} "
              f"(true ratio {29/nc:.4f}, i.e. {margin:.2f}x headroom)")
    manifest["k_ratio_star"] = ratio_star

    # ---------- 2. CVC-ClinicDB folds from the official table ----------
    print("\n=== CVC-ClinicDB: folds from the 29 official sequences ===")
    folds_c, loads_c = make_folds(truth_c, N_FOLDS)
    df_c = pd.DataFrame({"image": names_c, "frame": frames_c,
                         "group": truth_c, "fold": folds_c})
    df_c.to_csv(SPLITS / "cvc-clinicdb_groups_folds.csv", index=False)
    assert df_c.groupby("group")["fold"].nunique().max() == 1
    print(f"  29 groups -> fold image counts {'/'.join(map(str, loads_c))}; "
          f"0 sequences span folds")
    manifest["cvc-clinicdb"] = {"n_images": nc, "n_groups": 29,
                                "source": "official frame-range table"}

    # ---------- 3. Kvasir-SEG: uncertifiable, so report a sensitivity ladder ----------
    print("\n=== Kvasir-SEG: grouping strictness ladder (no order, no ground truth) ===")
    ii_k = np.flatnonzero(meta["dataset"].values == "kvasir-seg")
    names_k = meta.iloc[ii_k]["image"].values
    block_k = corr[np.ix_(ii_k, ii_k)]
    krows = []
    for tau in KVASIR_TAUS:
        adj = block_k > tau
        np.fill_diagonal(adj, False)
        n_comp, lab = connected_components(csr_matrix(adj), directed=False)
        sizes = pd.Series(lab).value_counts()
        giant = sizes.max() / len(ii_k)
        usable = giant <= GIANT_FRAC
        krows.append({"tau": tau, "n_groups": n_comp, "largest_group": int(sizes.max()),
                      "largest_frac": round(float(giant), 4),
                      "singletons": int((sizes == 1).sum()), "percolated": not usable})
        if usable:
            folds_k, loads_k = make_folds(lab, N_FOLDS)
            pd.DataFrame({"image": names_k, "group": lab, "fold": folds_k}).to_csv(
                SPLITS / f"kvasir-seg_groups_folds_tau{tau:.2f}.csv", index=False)
    kdf = pd.DataFrame(krows)
    print(kdf.to_string(index=False))
    kdf.to_csv(OUT / "kvasir_tau_ladder.csv", index=False)
    usable_taus = [r["tau"] for r in krows if not r["percolated"]]
    print(f"  usable thresholds (no giant component): {usable_taus}")
    print("  leak-freedom cannot be certified for Kvasir-SEG; the downstream experiment")
    print("  must report its estimate across these thresholds rather than at one.")
    manifest["kvasir-seg"] = {"n_images": int(len(ii_k)), "usable_taus": usable_taus,
                              "source": "similarity components; uncertified"}

    # ---------- 4. test-set keep-lists for the zero-shot datasets ----------
    print("\n=== zero-shot test sets: deduplicated keep-lists ===")
    dup = pd.read_csv(OUT / "exact_pixel_duplicates.csv")
    drop = {}
    for _, r in dup.iterrows():
        members = [m.split("/", 1) for m in r["images"].split(",")]
        keep = None
        for ds, img in members:
            if keep is None:
                keep = (ds, img)
            else:
                drop.setdefault(ds, set()).add(img)
    # CVC-300 is wholly contained in CVC-ColonDB, so it is not an independent set
    trows = []
    for name in ("cvc-colondb", "etis-larib", "cvc-300"):
        ii = np.flatnonzero(meta["dataset"].values == name)
        allnames = meta.iloc[ii]["image"].values
        dropped = drop.get(name, set())
        kept = [n for n in allnames if n not in dropped]
        pd.DataFrame({"image": kept}).to_csv(SPLITS / f"{name}_keep.csv", index=False)
        trows.append({"dataset": name, "n_images": len(allnames),
                      "exact_dupes_dropped": len(dropped), "n_kept": len(kept),
                      "independent_of_colondb": name != "cvc-300"})
    tdf = pd.DataFrame(trows)
    print(tdf.to_string(index=False))
    tdf.to_csv(SPLITS / "testset_keeplists.csv", index=False)
    print("  CVC-300 is a strict subset of CVC-ColonDB and must not be averaged as a")
    print("  separate dataset; its keep-list is emitted only for reproducing old numbers.")

    (SPLITS / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nsplits written to {SPLITS}")


if __name__ == "__main__":
    main()
