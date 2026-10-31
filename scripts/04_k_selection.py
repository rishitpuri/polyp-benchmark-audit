"""Choosing the number of sequences when no ground truth exists.

Script 03 showed that, given the true K, contiguous DP segmentation recovers
CVC-ClinicDB's 29 official sequences almost perfectly (ARI 0.997). But for
CVC-ColonDB, ETIS-Larib and CVC-300 the number of source videos was never
published, and the naive penalty used in script 03 saturated at the search
cap, making those applications meaningless.

Three selection rules are compared, all scored against CVC-ClinicDB's known
K=29:

  slope      Birge-Massart slope heuristic. Estimates the cost decrease per
             extra segment in the over-segmented regime and penalises at
             twice that rate. Uses no ground truth at all.
  elbow      Maximum curvature of the normalised cost curve. No ground truth.
  calibrated A penalty coefficient tuned so that CVC-ClinicDB yields K=29,
             then transferred. Uses ground truth, so it is only legitimate
             if it transfers; its stability is tested by halving ClinicDB
             and calibrating on one half to predict the other.

A rule is only applied to the unlabelled datasets if it recovers K=29 on
CVC-ClinicDB without having been told the answer.
"""

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
KMAX = 80


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
    """DP once; return cost-vs-K curve and backpointers for every K."""
    best = np.full((kmax + 1, n + 1), np.inf)
    back = np.zeros((kmax + 1, n + 1), dtype=int)
    best[0, 0] = 0.0
    for k in range(1, kmax + 1):
        for j in range(k, n + 1):
            cand = best[k - 1, :j] + cost[:j, j]
            i = int(np.argmin(cand))
            best[k, j], back[k, j] = cand[i], i
    return best[1:, n].copy(), back  # curve[i] is the cost for K = i + 1


def labels_for_k(back, n, k):
    bounds, j = [], n
    for kk in range(k, 0, -1):
        i = back[kk, j]
        bounds.append((i, j))
        j = i
    bounds.reverse()
    labels = np.zeros(n, dtype=int)
    for seq_id, (i, j) in enumerate(bounds, start=1):
        labels[i:j] = seq_id
    return labels


def select_slope(curve, kmax, frac=0.5):
    """Birge-Massart: fit the linear tail, penalise at twice the slope."""
    ks = np.arange(1, kmax + 1)
    lo = int(kmax * frac)
    slope = np.polyfit(ks[lo:], curve[lo:], 1)[0]
    kappa = -slope
    if kappa <= 0:
        return int(ks[int(np.argmin(curve))])
    return int(ks[int(np.argmin(curve + 2.0 * kappa * ks))])


def select_elbow(curve, kmax):
    """Point of maximum curvature on the min-max normalised curve."""
    ks = np.arange(1, kmax + 1, dtype=float)
    y = (curve - curve.min()) / (curve.max() - curve.min() + 1e-12)
    x = (ks - ks.min()) / (ks.max() - ks.min())
    # distance from each point to the chord joining the endpoints
    d = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return int(ks[int(np.argmax(d))])


def select_penalised(curve, kmax, beta):
    ks = np.arange(1, kmax + 1)
    return int(ks[int(np.argmin(curve / curve[0] + beta * ks))])


def true_labels_clinicdb(frames):
    lab = np.zeros(len(frames), dtype=int)
    for seq_id, (lo, hi) in enumerate(CLINICDB_SEQUENCES, start=1):
        lab[(frames >= lo) & (frames <= hi)] = seq_id
    return lab


def main():
    import re

    names, X = load("cvc-clinicdb")
    frames = np.array([int(re.sub(r"\D", "", n)) for n in names])
    truth = true_labels_clinicdb(frames)
    n = len(X)

    print(f"CVC-ClinicDB: n={n}, true K=29\n  running DP to K={KMAX} ...")
    cost = segment_costs(X)
    curve, back = dp_all(cost, n, KMAX)

    k_slope = select_slope(curve, KMAX)
    k_elbow = select_elbow(curve, KMAX)

    # beta that makes the calibrated rule return exactly 29
    betas = np.linspace(1e-5, 5e-2, 20000)
    picks = np.array([select_penalised(curve, KMAX, b) for b in betas])
    ok = betas[picks == 29]
    beta_star = float(np.median(ok)) if len(ok) else float("nan")

    print("\n=== K selection on CVC-ClinicDB (truth = 29) ===")
    rows = []
    for rule, k in (("slope (no GT)", k_slope), ("elbow (no GT)", k_elbow),
                    ("calibrated beta", 29 if len(ok) else -1)):
        if k > 0:
            ari = adjusted_rand_score(truth, labels_for_k(back, n, k))
            rows.append({"rule": rule, "K_selected": k, "ARI": round(ari, 4), "error": k - 29})
    print(pd.DataFrame(rows).to_string(index=False))
    if len(ok):
        print(f"\ncalibrated beta admissible range: [{ok.min():.5f}, {ok.max():.5f}] "
              f"(width {ok.max() - ok.min():.5f}), median {beta_star:.5f}")
    else:
        print("\nno beta yields K=29 -- calibration failed")

    # --- does the calibration generalise within ClinicDB? ---
    print("\n=== internal validation: calibrate on one half, test on the other ===")
    half = n // 2
    val_rows = []
    for tag, sl in (("first half", slice(0, half)), ("second half", slice(half, n))):
        Xh = X[sl]
        fh = frames[sl]
        th = truth[sl]
        k_true_h = len(np.unique(th))
        nh = len(Xh)
        kmax_h = min(KMAX, nh - 1)
        ch = segment_costs(Xh)
        curve_h, back_h = dp_all(ch, nh, kmax_h)
        bh = betas[np.array([select_penalised(curve_h, kmax_h, b) for b in betas]) == k_true_h]
        val_rows.append({
            "half": tag, "n": nh, "true_K": k_true_h,
            "slope": select_slope(curve_h, kmax_h),
            "elbow": select_elbow(curve_h, kmax_h),
            "beta_from_full_data": select_penalised(curve_h, kmax_h, beta_star) if ok.size else -1,
            "own_beta_range": f"[{bh.min():.4f},{bh.max():.4f}]" if len(bh) else "none",
        })
    print(pd.DataFrame(val_rows).to_string(index=False))

    pd.DataFrame({"K": np.arange(1, KMAX + 1), "cost": curve}).to_csv(
        OUT / "kselection_curve_clinicdb.csv", index=False)

    # --- apply whichever rules survived, to the unlabelled datasets ---
    print("\n=== applied to datasets with no published frame->video mapping ===")
    out_rows = []
    for name in ("cvc-colondb", "cvc-300", "etis-larib"):
        nm, Xd = load(name)
        nd = len(Xd)
        kmax_d = min(KMAX, nd - 1)
        cd = segment_costs(Xd)
        curve_d, back_d = dp_all(cd, nd, kmax_d)
        ks = {
            "slope": select_slope(curve_d, kmax_d),
            "elbow": select_elbow(curve_d, kmax_d),
            "calibrated": select_penalised(curve_d, kmax_d, beta_star) if ok.size else -1,
        }
        row = {"dataset": name, "n_frames": nd, **{f"K_{k}": v for k, v in ks.items()}}
        k_use = ks["calibrated"] if ks["calibrated"] > 0 else ks["slope"]
        lab = labels_for_k(back_d, nd, k_use)
        sizes = pd.Series(lab).value_counts()
        row.update({"K_used": k_use, "median_len": int(sizes.median()),
                    "min_len": int(sizes.min()), "max_len": int(sizes.max())})
        out_rows.append(row)
        pd.DataFrame({"image": nm, "sequence": lab}).to_csv(
            OUT / f"sequences_{name}.csv", index=False)
        pd.DataFrame({"K": np.arange(1, kmax_d + 1), "cost": curve_d}).to_csv(
            OUT / f"kselection_curve_{name}.csv", index=False)
    res = pd.DataFrame(out_rows)
    print(res.to_string(index=False))
    res.to_csv(OUT / "kselection_summary.csv", index=False)

    lab29 = labels_for_k(back, n, 29)
    pd.DataFrame({"image": names, "frame": frames, "sequence": lab29,
                  "official_sequence": truth}).to_csv(OUT / "sequences_cvc-clinicdb.csv", index=False)
    print(f"\nartifacts in {OUT}")


if __name__ == "__main__":
    main()
