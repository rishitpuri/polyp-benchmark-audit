"""Image-level contamination audit across public polyp segmentation datasets.

Computes, for every ordered pair of datasets, how many images in A have a
near-duplicate in B. Two independent signals are used so that neither alone
drives a conclusion:

  thumb   normalised 64x64 greyscale thumbnail, compared by Pearson r.
          Resolution-invariant, sensitive to photometric re-encoding.
  phash   perceptual hash (16x16 DCT, 256 bits), compared by Hamming
          distance. Robust to mild cropping and compression.

An image pair is called an exact-frame match only when both signals agree.
Everything above the reporting threshold is written out with both paths so
that matches can be inspected by eye before any claim is made.
"""

import hashlib
import itertools
import json
import sys
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "contamination"

# The five datasets of the standard evaluation suite, plus Kvasir-Sessile,
# which is distributed separately and is sometimes used as a held-out set.
# Kvasir-Sessile is NOT part of the suite total: it is drawn from Kvasir-SEG,
# so counting it there would double-count its images.
SUITE = ["kvasir-seg", "cvc-clinicdb", "cvc-colondb", "cvc-300", "etis-larib"]
DATASETS = {
    "kvasir-seg": DATA / "Kvasir-SEG" / "images",
    "kvasir-sessile": DATA / "sessile-main-Kvasir-SEG/images",
    "cvc-clinicdb": DATA / "external/cvc-clinicdb/images",
    "cvc-colondb": DATA / "external/cvc-colondb/images",
    "cvc-300": DATA / "external/cvc-300/images",
    "etis-larib": DATA / "external/etis-larib/images",
}

THUMB = 64
PHASH_SIZE = 16
R_THRESHOLDS = [0.999, 0.99, 0.98, 0.95, 0.90]
HAMMING_THRESHOLDS = [0, 2, 5, 10, 20]
REPORT_R = 0.90


def load_features(name, directory):
    rows = []
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    if not paths:
        sys.exit(f"no images found for {name} in {directory}")
    for path in paths:
        with Image.open(path) as im:
            width, height = im.size
            rgb = im.convert("RGB")
            grey = rgb.convert("L").resize((THUMB, THUMB), Image.BILINEAR)
            vec = np.asarray(grey, dtype=np.float64).ravel()
            vec = vec - vec.mean()
            norm = np.linalg.norm(vec)
            rows.append(
                {
                    "dataset": name,
                    "image": path.name,
                    "path": str(path),
                    "width": width,
                    "height": height,
                    "pixel_md5": hashlib.md5(np.asarray(rgb, dtype=np.uint8).tobytes()).hexdigest(),
                    "phash": str(imagehash.phash(rgb, hash_size=PHASH_SIZE)),
                    "_vec": vec / norm if norm > 0 else vec,
                }
            )
    print(f"  {name:<14} {len(rows):>5} images  {rows[0]['width']}x{rows[0]['height']}")
    return rows


def phash_bits(hex_strings):
    """Unpack hex phash strings into a (n, nbits) uint8 bit matrix."""
    nbits = PHASH_SIZE * PHASH_SIZE
    out = np.zeros((len(hex_strings), nbits), dtype=np.uint8)
    for i, hex_string in enumerate(hex_strings):
        value = int(hex_string, 16)
        bits = np.array([(value >> k) & 1 for k in range(nbits)], dtype=np.uint8)
        out[i] = bits[::-1]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("loading datasets")
    rows = []
    for name, directory in DATASETS.items():
        if not directory.exists():
            sys.exit(f"missing directory for {name}: {directory}")
        rows.extend(load_features(name, directory))

    vecs = np.stack([r.pop("_vec") for r in rows])
    meta = pd.DataFrame(rows)
    bits = phash_bits(meta["phash"].tolist())
    n_suite = int(meta["dataset"].isin(SUITE).sum())
    print(f"  suite total    {n_suite:>5} images "
          f"(+{len(meta) - n_suite} in separately distributed sets)\n")

    # Pearson r between all thumbnails == dot product of centred unit vectors.
    corr = vecs @ vecs.T
    hamming = (bits[:, None, :] != bits[None, :, :]).sum(-1) if len(meta) < 3000 else None
    if hamming is None:
        sys.exit("too many images for the dense Hamming matrix; chunk it")

    idx = {name: np.flatnonzero(meta["dataset"].values == name) for name in DATASETS}

    # ---- exact pixel duplicates, within and across datasets -------------
    dup_groups = [g for _, g in meta.groupby("pixel_md5").groups.items() if len(g) > 1]
    exact = []
    for group in dup_groups:
        members = meta.loc[group]
        exact.append(
            {
                "n": len(members),
                "datasets": ",".join(sorted(members["dataset"].unique())),
                "images": ",".join(f"{d}/{i}" for d, i in zip(members["dataset"], members["image"])),
            }
        )
    pd.DataFrame(exact).to_csv(OUT / "exact_pixel_duplicates.csv", index=False)
    cross_exact = [e for e in exact if "," in e["datasets"]]
    print(f"exact pixel-identical groups: {len(exact)} total, {len(cross_exact)} spanning >1 dataset")

    # ---- pairwise cross-dataset near-duplicate counts -------------------
    summary = []
    matches = []
    for a, b in itertools.permutations(DATASETS, 2):
        ia, ib = idx[a], idx[b]
        sub_r = corr[np.ix_(ia, ib)]
        sub_h = hamming[np.ix_(ia, ib)]
        best = sub_r.argmax(1)
        best_r = sub_r[np.arange(len(ia)), best]
        best_h = sub_h[np.arange(len(ia)), best]

        record = {"query": a, "reference": b, "n_query": len(ia), "n_reference": len(ib)}
        for t in R_THRESHOLDS:
            record[f"n_r>{t}"] = int((best_r > t).sum())
        for t in HAMMING_THRESHOLDS:
            record[f"n_hamming<={t}"] = int((sub_h.min(1) <= t).sum())
        record["n_both"] = int(((best_r > 0.98) & (best_h <= 10)).sum())
        summary.append(record)

        for qi in np.flatnonzero(best_r > REPORT_R):
            matches.append(
                {
                    "query_dataset": a,
                    "query_image": meta.iloc[ia[qi]]["image"],
                    "ref_dataset": b,
                    "ref_image": meta.iloc[ib[best[qi]]]["image"],
                    "pearson_r": round(float(best_r[qi]), 5),
                    "phash_hamming": int(best_h[qi]),
                    "query_path": meta.iloc[ia[qi]]["path"],
                    "ref_path": meta.iloc[ib[best[qi]]]["path"],
                }
            )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "cross_dataset_summary.csv", index=False)
    matches_df = pd.DataFrame(matches).sort_values("pearson_r", ascending=False)
    matches_df.to_csv(OUT / "cross_dataset_matches.csv", index=False)

    # ---- within-dataset redundancy (proxy for sequence structure) -------
    within = []
    for name, ii in idx.items():
        sub_r = corr[np.ix_(ii, ii)].copy()
        np.fill_diagonal(sub_r, -np.inf)
        best_r = sub_r.max(1)
        within.append(
            {
                "dataset": name,
                "n": len(ii),
                "median_nn_r": round(float(np.median(best_r)), 4),
                "frac_nn_r>0.98": round(float((best_r > 0.98).mean()), 4),
                "frac_nn_r>0.95": round(float((best_r > 0.95).mean()), 4),
                "frac_nn_r>0.90": round(float((best_r > 0.90).mean()), 4),
            }
        )
    within_df = pd.DataFrame(within)
    within_df.to_csv(OUT / "within_dataset_redundancy.csv", index=False)

    np.save(OUT / "thumb_corr.npy", corr.astype(np.float32))
    np.save(OUT / "phash_hamming.npy", hamming.astype(np.int16))
    meta.drop(columns=[]).to_csv(OUT / "image_index.csv", index=False)

    print("\n=== cross-dataset nearest-neighbour counts (query -> reference) ===")
    cols = ["query", "reference", "n_query", "n_r>0.999", "n_r>0.99", "n_r>0.98", "n_r>0.95", "n_hamming<=0", "n_hamming<=5", "n_both"]
    print(summary_df[cols].to_string(index=False))

    print("\n=== within-dataset redundancy ===")
    print(within_df.to_string(index=False))

    print(f"\n=== strongest cross-dataset matches (r > {REPORT_R}) ===")
    if matches_df.empty:
        print("none")
    else:
        print(matches_df.head(25)[["query_dataset", "query_image", "ref_dataset", "ref_image", "pearson_r", "phash_hamming"]].to_string(index=False))
        print(f"\n{len(matches_df)} matches written to {OUT / 'cross_dataset_matches.csv'}")

    (OUT / "config.json").write_text(
        json.dumps(
            {
                "thumb": THUMB,
                "phash_size": PHASH_SIZE,
                "r_thresholds": R_THRESHOLDS,
                "hamming_thresholds": HAMMING_THRESHOLDS,
                "datasets": {k: str(v) for k, v in DATASETS.items()},
            },
            indent=2,
        )
    )
    print(f"\nartifacts in {OUT}")


if __name__ == "__main__":
    main()
