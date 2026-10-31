"""Fixed-point slope heuristic for choosing the number of sequences.

Script 04 found the Birge-Massart slope heuristic recovers K=32 on
CVC-ClinicDB (true 29, ARI 0.975) but K=27 and K=25 on its two halves (true
15 each). That instability is an artefact of the search cap: the tail used to
estimate the cost-per-segment was fixed at half of KMAX=80, which sits at
2.8x the true K for the full dataset but 5.3x for the halves, so the two fits
sample quite different parts of the curve.

The fix is to let the search cap follow the estimate. Starting from a loose
cap, estimate K, reset the cap to a fixed multiple of that estimate, and
repeat to a fixed point. The tail window then occupies the same relative
position on the curve regardless of dataset size.

Validation uses two anchors, not one:
  CVC-ClinicDB   29 sequences, from the official frame-range table.
  ETIS-Larib     34 sequences, per the dataset's published description.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

OUT = Path(__file__).resolve().parent.parent / "outputs" / "contamination"
CACHE = OUT / "features"

CLINICDB_SEQUENCES = [
    (1, 25), (26, 50), (51, 67), (68, 78), (79, 103), (104, 126), (127, 151),
    (152, 177), (178, 199), (200, 205), (206, 227), (228, 252), (253, 277),
    (278, 297), (298, 317), (318, 342), (343, 363), (364, 383), (384, 408),
    (409, 428), (429, 447), (448, 466), (467, 478), (479, 503), (504, 528),
    (529, 546), (547, 571), (572, 591), (592, 612),
]
DOCUMENTED_K = {"cvc-clinicdb": 29, "etis-larib": 34}
CAP_MULTIPLE = 3.0
TAIL_FRAC = 0.5


def load(name):
    d = np.load(CACHE / f"{name}.npz", allow_pickle=True)
    return list(d["images"]), d["X"]


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


def dp_all(cost, n, kmax):
    best = np.full((kmax + 1, n + 1), np.inf)
    back = np.zeros((kmax + 1, n + 1), dtype=int)
    best[0, 0] = 0.0
    for k in range(1, kmax + 1):
        for j in range(k, n + 1):
            cand = best[k - 1, :j] + cost[:j, j]
            i = int(np.argmin(cand))
            best[k, j], back[k, j] = cand[i], i
    return best[1:, n].copy(), back


def labels_for_k(back, n, k):
    bounds, j = [], n
    for kk in range(k, 0, -1):
        i = back[kk, j]
        bounds.append((i, j))
        j = i
    labels = np.zeros(n, dtype=int)
    for seq_id, (i, j) in enumerate(reversed(bounds), start=1):
        labels[i:j] = seq_id
    return labels


def slope_k(curve, kmax, tail_frac=TAIL_FRAC):
    ks = np.arange(1, kmax + 1)
    lo = max(1, int(kmax * tail_frac))
    if kmax - lo < 3:
        return int(ks[int(np.argmin(curve))])
    kappa = -np.polyfit(ks[lo:], curve[lo:], 1)[0]
    if kappa <= 0:
        return int(ks[int(np.argmin(curve))])
    return int(ks[int(np.argmin(curve + 2.0 * kappa * ks))])


def fixed_point_k(cost, n, k_init=None, iters=6):
    """Iterate cap <- CAP_MULTIPLE * K_hat until the estimate stops moving."""
    kmax = min(n - 1, int(k_init or max(20, n // 4)))
    history = []
    for _ in range(iters):
        curve, back = dp_all(cost, n, kmax)
        k_hat = slope_k(curve, kmax)
        history.append((kmax, k_hat))
        new_cap = int(np.clip(round(CAP_MULTIPLE * k_hat), 6, n - 1))
        if new_cap == kmax:
            break
        kmax = new_cap
    curve, back = dp_all(cost, n, kmax)
    return slope_k(curve, kmax), kmax, back, curve, history


def true_labels_clinicdb(frames):
    lab = np.zeros(len(frames), dtype=int)
    for i, (lo, hi) in enumerate(CLINICDB_SEQUENCES, start=1):
        lab[(frames >= lo) & (frames <= hi)] = i
    return lab


def main():
    names, X = load("cvc-clinicdb")
    frames = np.array([int(re.sub(r"\D", "", s)) for s in names])
    truth = true_labels_clinicdb(frames)
    n = len(X)
    cost = segment_costs(X)

    print("=== CVC-ClinicDB (true K=29) ===")
    k_hat, kmax, back, curve, hist = fixed_point_k(cost, n)
    ari = adjusted_rand_score(truth, labels_for_k(back, n, k_hat))
    print(f"  fixed-point iterations (cap -> K_hat): {hist}")
    print(f"  K_hat = {k_hat}  (error {k_hat - 29:+d}),  ARI {ari:.4f},  final cap {kmax}")

    print("\n=== internal validation on contiguous halves (true K=15 each) ===")
    rows = []
    half = n // 2
    for tag, sl in (("first", slice(0, half)), ("second", slice(half, n))):
        Xh, th = X[sl], truth[sl]
        nh, k_true = len(Xh), len(np.unique(th))
        ch = segment_costs(Xh)
        kh, caph, backh, _, histh = fixed_point_k(ch, nh)
        rows.append({"half": tag, "n": nh, "true_K": k_true, "K_hat": kh,
                     "error": kh - k_true,
                     "ARI": round(adjusted_rand_score(th, labels_for_k(backh, nh, kh)), 4),
                     "final_cap": caph})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== applied to all datasets ===")
    out = []
    for name in ("cvc-clinicdb", "cvc-colondb", "cvc-300", "etis-larib"):
        nm, Xd = load(name)
        nd = len(Xd)
        cd = segment_costs(Xd)
        kd, capd, backd, curved, _ = fixed_point_k(cd, nd)
        lab = labels_for_k(backd, nd, kd)
        sizes = pd.Series(lab).value_counts()
        row = {"dataset": name, "n_frames": nd, "K_hat": kd,
               "documented_K": DOCUMENTED_K.get(name, None),
               "median_len": int(sizes.median()), "min_len": int(sizes.min()),
               "max_len": int(sizes.max())}
        if name in DOCUMENTED_K:
            row["error"] = kd - DOCUMENTED_K[name]
        out.append(row)
        pd.DataFrame({"image": nm, "sequence": lab}).to_csv(
            OUT / f"final_sequences_{name}.csv", index=False)
    res = pd.DataFrame(out)
    print(res.to_string(index=False))
    res.to_csv(OUT / "final_sequence_summary.csv", index=False)
    print(f"\nartifacts in {OUT}")


if __name__ == "__main__":
    main()
