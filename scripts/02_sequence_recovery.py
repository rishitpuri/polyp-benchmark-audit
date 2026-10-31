"""Can video-sequence membership be recovered from pixels alone?

Most public polyp datasets never released a frame -> source-video mapping, so
sequence-disjoint splits cannot be built for them. CVC-ClinicDB is the one
exception: its README gives the official frame-range -> sequence table for all
612 frames and 29 sequences.

That makes CVC-ClinicDB a validation set for the recovery problem itself. Here
we cluster frames using only the thumbnail correlation matrix from script 01
and score the clustering against the official sequence labels. If recovery
works on CVC-ClinicDB, the same procedure can be applied to CVC-ColonDB,
ETIS-Larib and Kvasir-SEG, where no ground truth exists.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

OUT = Path(__file__).resolve().parent.parent / "outputs" / "contamination"

# Official CVC-ClinicDB frame-range -> sequence table (29 sequences, 612 frames).
CLINICDB_SEQUENCES = [
    (1, 25), (26, 50), (51, 67), (68, 78), (79, 103), (104, 126), (127, 151),
    (152, 177), (178, 199), (200, 205), (206, 227), (228, 252), (253, 277),
    (278, 297), (298, 317), (318, 342), (343, 363), (364, 383), (384, 408),
    (409, 428), (429, 447), (448, 466), (467, 478), (479, 503), (504, 528),
    (529, 546), (547, 571), (572, 591), (592, 612),
]


def true_sequence(frame_number):
    for seq_id, (lo, hi) in enumerate(CLINICDB_SEQUENCES, start=1):
        if lo <= frame_number <= hi:
            return seq_id
    raise ValueError(f"frame {frame_number} outside the official table")


def cluster(corr_block, threshold, method="average"):
    """Agglomerative clustering on correlation distance."""
    dist = np.clip(1.0 - corr_block, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    link = linkage(squareform(dist, checks=False), method=method)
    return fcluster(link, t=threshold, criterion="distance")


def main():
    meta = pd.read_csv(OUT / "image_index.csv")
    corr = np.load(OUT / "thumb_corr.npy").astype(np.float64)

    idx = np.flatnonzero(meta["dataset"].values == "cvc-clinicdb")
    sub = meta.iloc[idx].copy()
    sub["frame"] = [int(re.sub(r"\D", "", n)) for n in sub["image"]]
    order = np.argsort(sub["frame"].values)
    idx, sub = idx[order], sub.iloc[order]
    sub["true_seq"] = [true_sequence(f) for f in sub["frame"]]
    block = corr[np.ix_(idx, idx)]

    print(f"CVC-ClinicDB: {len(sub)} frames, {sub.true_seq.nunique()} official sequences")
    print(f"  official sequence sizes: min {sub.true_seq.value_counts().min()}, "
          f"max {sub.true_seq.value_counts().max()}, median {int(sub.true_seq.value_counts().median())}\n")

    rows = []
    for method in ("average", "complete", "ward"):
        for threshold in (0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30):
            labels = cluster(block, threshold, method)
            ari = adjusted_rand_score(sub["true_seq"], labels)
            hom, comp, vms = homogeneity_completeness_v_measure(sub["true_seq"], labels)
            rows.append({
                "method": method, "threshold": threshold, "n_clusters": len(set(labels)),
                "ARI": round(ari, 4), "homogeneity": round(hom, 4),
                "completeness": round(comp, 4), "v_measure": round(vms, 4),
            })
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "sequence_recovery_clinicdb.csv", index=False)
    print("=== recovery of the 29 official sequences from pixels alone ===")
    print(results.to_string(index=False))

    best = results.loc[results["ARI"].idxmax()]
    print(f"\nbest: {best['method']} @ {best['threshold']} -> "
          f"{int(best['n_clusters'])} clusters, ARI {best['ARI']}, V {best['v_measure']}")

    # How badly does a random split leak, given the official sequences?
    rng = np.random.default_rng(0)
    leak_random, leak_seq = [], []
    for _ in range(1000):
        perm = rng.permutation(len(sub))
        test = perm[: len(sub) // 5]
        train = perm[len(sub) // 5:]
        shared = len(set(sub["true_seq"].values[test]) & set(sub["true_seq"].values[train]))
        leak_random.append(shared / len(set(sub["true_seq"].values[test])))

        seqs = rng.permutation(sorted(sub["true_seq"].unique()))
        test_seqs = set(seqs[: max(1, len(seqs) // 5)])
        in_test = sub["true_seq"].isin(test_seqs).values
        leak_seq.append(len(set(sub["true_seq"].values[in_test]) & set(sub["true_seq"].values[~in_test]))
                        / len(test_seqs))

    print(f"\n=== fraction of test sequences also present in train (1000 draws) ===")
    print(f"  random 80/20 frame split   : {np.mean(leak_random):.3f}  "
          f"(min {np.min(leak_random):.3f}, max {np.max(leak_random):.3f})")
    print(f"  sequence-disjoint 80/20    : {np.mean(leak_seq):.3f}")

    # Apply the validated setting to datasets with no published sequence table.
    best_method, best_t = best["method"], float(best["threshold"])
    print(f"\n=== reconstructed sequence counts using {best_method} @ {best_t} ===")
    recon = []
    for name in ("kvasir-seg", "cvc-colondb", "cvc-300", "etis-larib"):
        ii = np.flatnonzero(meta["dataset"].values == name)
        labels = cluster(corr[np.ix_(ii, ii)], best_t, best_method)
        sizes = pd.Series(labels).value_counts()
        recon.append({
            "dataset": name, "n_images": len(ii), "n_clusters": len(set(labels)),
            "largest_cluster": int(sizes.max()), "singletons": int((sizes == 1).sum()),
            "median_cluster": int(sizes.median()),
        })
        pd.DataFrame({"image": meta.iloc[ii]["image"].values, "cluster": labels}).to_csv(
            OUT / f"recovered_sequences_{name}.csv", index=False)
    recon_df = pd.DataFrame(recon)
    print(recon_df.to_string(index=False))
    recon_df.to_csv(OUT / "recovered_sequence_summary.csv", index=False)

    pd.DataFrame({
        "image": sub["image"].values, "frame": sub["frame"].values,
        "official_sequence": sub["true_seq"].values,
        "recovered_cluster": cluster(block, best_t, best_method),
    }).to_csv(OUT / "clinicdb_sequence_labels.csv", index=False)
    print(f"\nartifacts in {OUT}")


if __name__ == "__main__":
    main()
