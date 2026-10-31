"""Recover video-sequence structure by change-point detection on ordered frames.

Script 02 treated sequence recovery as unordered clustering and reached only
ARI 0.50 against CVC-ClinicDB's 29 official sequences. That discards the
strongest available signal: in CVC-ClinicDB the official sequences are
*contiguous* frame ranges, so recovery is a segmentation problem on an ordered
stream, not a clustering problem.

Here we solve the exact problem: partition frames 1..n into K contiguous
segments minimising within-segment feature variance, by dynamic programming
(optimal, not greedy). CVC-ClinicDB supplies ground truth, so the method can
be validated before being applied to CVC-ColonDB, CVC-300 and ETIS-Larib,
which never published a frame-to-video mapping.

Kvasir-SEG is excluded: its filenames are content hashes carrying no temporal
order, so no ordered method applies and clustering remains the only option.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "contamination"
CACHE = OUT / "features"

ORDERED_DATASETS = {
    "cvc-clinicdb": DATA / "external/cvc-clinicdb/images",
    "cvc-colondb": DATA / "external/cvc-colondb/images",
    "cvc-300": DATA / "external/cvc-300/images",
    "etis-larib": DATA / "external/etis-larib/images",
}

CLINICDB_SEQUENCES = [
    (1, 25), (26, 50), (51, 67), (68, 78), (79, 103), (104, 126), (127, 151),
    (152, 177), (178, 199), (200, 205), (206, 227), (228, 252), (253, 277),
    (278, 297), (298, 317), (318, 342), (343, 363), (364, 383), (384, 408),
    (409, 428), (429, 447), (448, 466), (467, 478), (479, 503), (504, 528),
    (529, 546), (547, 571), (572, 591), (592, 612),
]

THUMB = 32
HIST_BINS = 16


def frame_number(name):
    digits = re.sub(r"\D", "", name)
    return int(digits) if digits else -1


def features(name, directory):
    cache = CACHE / f"{name}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return list(d["images"]), d["X"]

    paths = sorted(
        (p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}),
        key=lambda p: frame_number(p.name),
    )
    rows, names = [], []
    for path in paths:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            grey = np.asarray(rgb.convert("L").resize((THUMB, THUMB), Image.BILINEAR), dtype=np.float64).ravel()
            grey = grey / 255.0
            arr = np.asarray(rgb.resize((64, 64), Image.BILINEAR), dtype=np.float64) / 255.0
            hist = np.concatenate([
                np.histogram(arr[..., c], bins=HIST_BINS, range=(0, 1), density=True)[0]
                for c in range(3)
            ])
        # Both blocks L2-normalised so neither dominates the variance cost.
        grey = grey / (np.linalg.norm(grey) + 1e-12)
        hist = hist / (np.linalg.norm(hist) + 1e-12)
        rows.append(np.concatenate([grey, hist]))
        names.append(path.name)
    X = np.stack(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, images=np.array(names), X=X)
    return names, X


def segment_costs(X):
    """cost[i, j] = within-segment sum of squared deviations for frames [i, j)."""
    n = len(X)
    s1 = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, axis=0)])
    s2 = np.concatenate([[0.0], np.cumsum((X ** 2).sum(1))])
    cost = np.full((n + 1, n + 1), np.inf)
    for i in range(n):
        j = np.arange(i + 1, n + 1)
        length = j - i
        block = s1[j] - s1[i]
        cost[i, j] = (s2[j] - s2[i]) - (block ** 2).sum(1) / length
    return np.maximum(cost, 0.0)


def dp_segment(cost, n, k):
    """Optimal partition of 0..n into exactly k contiguous segments."""
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
    bounds.reverse()
    labels = np.zeros(n, dtype=int)
    for seq_id, (i, j) in enumerate(bounds, start=1):
        labels[i:j] = seq_id
    return labels, best[k, n], [b[0] for b in bounds[1:]]


def boundary_scores(pred, true, tol):
    hit = sum(any(abs(p - t) <= tol for p in pred) for t in true)
    recall = hit / len(true) if true else 0.0
    prec = sum(any(abs(p - t) <= tol for t in true) for p in pred) / len(pred) if pred else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
    return round(prec, 4), round(recall, 4), round(f1, 4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    names, X = features("cvc-clinicdb", ORDERED_DATASETS["cvc-clinicdb"])
    frames = np.array([frame_number(n) for n in names])
    true_labels = np.zeros(len(frames), dtype=int)
    for seq_id, (lo, hi) in enumerate(CLINICDB_SEQUENCES, start=1):
        true_labels[(frames >= lo) & (frames <= hi)] = seq_id
    true_bounds = [int(np.searchsorted(frames, lo)) for lo, _ in CLINICDB_SEQUENCES[1:]]

    print(f"CVC-ClinicDB: {len(frames)} frames in order, {len(CLINICDB_SEQUENCES)} official sequences")
    print(f"feature dim {X.shape[1]}\n")

    cost = segment_costs(X)
    n = len(X)

    # --- baseline: consecutive-frame dissimilarity, take the K-1 largest gaps
    gaps = np.linalg.norm(np.diff(X, axis=0), axis=1)
    k_true = len(CLINICDB_SEQUENCES)
    naive_bounds = sorted((np.argsort(gaps)[-(k_true - 1):] + 1).tolist())
    naive_labels = np.cumsum(np.isin(np.arange(n), naive_bounds)) + 1
    print("=== baseline: K-1 largest consecutive-frame gaps ===")
    print(f"  ARI {adjusted_rand_score(true_labels, naive_labels):.4f}   "
          f"boundary F1 (+/-2 frames) {boundary_scores(naive_bounds, true_bounds, 2)[2]}")

    # --- optimal DP segmentation, sweeping K
    print("\n=== optimal DP contiguous segmentation ===")
    rows = []
    for k in [10, 20, 25, 29, 32, 35, 40, 50, 60]:
        labels, total, bounds = dp_segment(cost, n, k)
        ari = adjusted_rand_score(true_labels, labels)
        hom, comp, vms = homogeneity_completeness_v_measure(true_labels, labels)
        rows.append({
            "K": k, "cost": round(float(total), 4), "ARI": round(ari, 4),
            "homogeneity": round(hom, 4), "completeness": round(comp, 4), "v_measure": round(vms, 4),
            "bF1_tol0": boundary_scores(bounds, true_bounds, 0)[2],
            "bF1_tol1": boundary_scores(bounds, true_bounds, 1)[2],
            "bF1_tol2": boundary_scores(bounds, true_bounds, 2)[2],
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "changepoint_clinicdb.csv", index=False)
    print(res.to_string(index=False))

    at_true = res[res.K == k_true].iloc[0]
    print(f"\nat the true K=29: ARI {at_true.ARI}, V {at_true.v_measure}, "
          f"boundary F1 {at_true.bF1_tol0} (exact) / {at_true.bF1_tol2} (+/-2 frames)")

    # --- K selection without ground truth, for the other datasets
    print("\n=== automatic K selection (elbow on penalised cost) ===")
    ks = np.arange(5, 61)
    costs = np.array([dp_segment(cost, n, int(k))[1] for k in ks])
    pen = costs + 0.02 * ks * np.log(n) * costs[-1] / n
    k_auto = int(ks[int(np.argmin(pen))])
    labels_auto, _, _ = dp_segment(cost, n, k_auto)
    print(f"  selected K={k_auto} (true 29); ARI {adjusted_rand_score(true_labels, labels_auto):.4f}")
    pd.DataFrame({"K": ks, "cost": costs, "penalised": pen}).to_csv(OUT / "changepoint_kcurve.csv", index=False)

    best_labels, _, _ = dp_segment(cost, n, k_true)
    pd.DataFrame({
        "image": names, "frame": frames,
        "official_sequence": true_labels, "recovered_sequence": best_labels,
    }).to_csv(OUT / "clinicdb_changepoint_labels.csv", index=False)

    # --- apply to datasets with no published mapping
    print("\n=== applied to datasets with no published frame->video mapping ===")
    summary = []
    for name, directory in ORDERED_DATASETS.items():
        if name == "cvc-clinicdb":
            continue
        nm, Xd = features(name, directory)
        cd = segment_costs(Xd)
        nd = len(Xd)
        ks_d = np.arange(2, min(61, nd))
        costs_d = np.array([dp_segment(cd, nd, int(k))[1] for k in ks_d])
        pen_d = costs_d + 0.02 * ks_d * np.log(nd) * costs_d[-1] / nd
        k_sel = int(ks_d[int(np.argmin(pen_d))])
        labels_d, _, _ = dp_segment(cd, nd, k_sel)
        sizes = pd.Series(labels_d).value_counts()
        summary.append({
            "dataset": name, "n_frames": nd, "K_selected": k_sel,
            "median_seq_len": int(sizes.median()), "min_seq_len": int(sizes.min()),
            "max_seq_len": int(sizes.max()),
        })
        pd.DataFrame({"image": nm, "frame": [frame_number(x) for x in nm],
                      "recovered_sequence": labels_d}).to_csv(
            OUT / f"changepoint_sequences_{name}.csv", index=False)
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(index=False))
    sdf.to_csv(OUT / "changepoint_summary.csv", index=False)
    print(f"\nartifacts in {OUT}")


if __name__ == "__main__":
    main()
