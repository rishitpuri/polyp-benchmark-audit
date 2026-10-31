"""Re-score the standard five-dataset benchmark with the duplication removed.

Uses the per-image metrics from the 88 training runs of paper 1 (five
architectures x five folds, each evaluated on Kvasir-SEG and zero-shot on four
external sets), so no retraining is needed. Two corrections are applied:

  1. CVC-300 is dropped. All 60 of its images are pixel-identical to images in
     CVC-ColonDB (frames 149-208), so averaging it as a separate dataset
     counts those images twice.
  2. Exact duplicate images are collapsed to one instance each: 34 of the 380
     CVC-ColonDB files and 3 of the 196 ETIS-Larib files are redundant, which
     silently up-weights the lesions they contain.

The question is not only whether the mean moves, but whether it moves by more
than the run-to-run noise floor of the protocol, sigma_run = 0.0053 Dice,
measured in paper 1 -- and whether the ranking of architectures changes.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "outputs" / "splits"
OUT = ROOT / "outputs" / "rescore"

SIGMA_RUN = 0.0053
EXTERNAL_PUBLISHED = ["cvc-clinicdb", "cvc-colondb", "cvc-300", "etis-larib"]
EXTERNAL_CORRECTED = ["cvc-clinicdb", "cvc-colondb", "etis-larib"]
METRIC = "dice"


def load_perimage():
    files = [f for f in sorted(glob.glob(str(ROOT / "paper1_results/outputs/cv2/perimage_*.csv")))
             if not f.endswith("_tta.csv")]
    if not files:
        raise SystemExit("no per-image CSVs found")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df, len(files)


def keep_mask(df):
    """True for rows that survive deduplication."""
    mask = pd.Series(True, index=df.index)
    for ds in ("cvc-colondb", "etis-larib"):
        path = SPLITS / f"{ds}_keep.csv"
        kept = set(pd.read_csv(path)["image"].astype(str))
        sel = df["dataset"] == ds
        mask.loc[sel] = df.loc[sel, "name"].astype(str).isin(kept)
    return mask


def dataset_means(df, metric=METRIC):
    return df.groupby(["model", "dataset"])[metric].mean().unstack()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, nfiles = load_perimage()
    print(f"loaded {nfiles} per-image files, {len(df)} rows, "
          f"{df.model.nunique()} architectures\n")

    # names in the CSVs may carry extensions; normalise against the keep-lists
    df["name"] = df["name"].astype(str)
    sample = df[df.dataset == "cvc-colondb"]["name"].iloc[0]
    kept_sample = pd.read_csv(SPLITS / "cvc-colondb_keep.csv")["image"].iloc[0]
    if "." in str(kept_sample) and "." not in sample:
        df.loc[df.dataset.isin(["cvc-colondb", "etis-larib"]), "name"] += ".png"
    elif "." not in str(kept_sample) and "." in sample:
        df["name"] = df["name"].str.rsplit(".", n=1).str[0]

    mask = keep_mask(df)
    dropped = (~mask).sum()
    print(f"deduplication removes {dropped} rows "
          f"({dropped // df.model.nunique()} per architecture)")
    if dropped == 0:
        raise SystemExit("keep-list join matched nothing -- check filename formats")

    pub = dataset_means(df)
    cor = dataset_means(df[mask])

    # --- per-dataset effect of collapsing duplicates ---
    print("\n=== per-dataset mean Dice: published vs deduplicated ===")
    rows = []
    for ds in ("cvc-colondb", "etis-larib"):
        for model in sorted(df.model.unique()):
            a, b = pub.loc[model, ds], cor.loc[model, ds]
            rows.append({"dataset": ds, "model": model, "published": round(a, 4),
                         "dedup": round(b, 4), "delta": round(b - a, 4),
                         "exceeds_sigma": abs(b - a) > SIGMA_RUN})
    perds = pd.DataFrame(rows)
    print(perds.to_string(index=False))
    perds.to_csv(OUT / "per_dataset_dedup_effect.csv", index=False)

    # --- the headline: the averaged "external" score ---
    print("\n=== averaged external score: published 4-dataset vs corrected 3-dataset ===")
    summary = []
    for model in sorted(df.model.unique()):
        a = float(np.mean([pub.loc[model, d] for d in EXTERNAL_PUBLISHED]))
        b = float(np.mean([cor.loc[model, d] for d in EXTERNAL_CORRECTED]))
        summary.append({"model": model, "published_avg4": round(a, 4),
                        "corrected_avg3": round(b, 4), "delta": round(b - a, 4),
                        "delta_in_sigma": round((b - a) / SIGMA_RUN, 2)})
    s = pd.DataFrame(summary).sort_values("published_avg4", ascending=False)
    s["rank_published"] = range(1, len(s) + 1)
    s = s.sort_values("corrected_avg3", ascending=False)
    s["rank_corrected"] = range(1, len(s) + 1)
    s["rank_change"] = s["rank_published"] - s["rank_corrected"]
    print(s.to_string(index=False))
    s.to_csv(OUT / "external_average_shift.csv", index=False)

    print(f"\n  sigma_run = {SIGMA_RUN}")
    print(f"  largest |delta| = {s.delta.abs().max():.4f} "
          f"({s.delta.abs().max()/SIGMA_RUN:.1f} sigma_run)")
    print(f"  architectures whose shift exceeds the noise floor: "
          f"{int((s.delta.abs() > SIGMA_RUN).sum())} of {len(s)}")
    if (s["rank_change"] != 0).any():
        print("  RANKING CHANGES under the correction:")
        for _, r in s[s.rank_change != 0].iterrows():
            print(f"    {r.model}: rank {r.rank_published} -> {r.rank_corrected}")
    else:
        print("  ranking is unchanged by the correction")

    # --- pairwise gaps vs the noise floor, before and after ---
    print("\n=== pairwise architecture gaps on the external average ===")
    models = list(s.model)
    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            mi, mj = models[i], models[j]
            gp = float(s.loc[s.model == mi, "published_avg4"].iloc[0]
                       - s.loc[s.model == mj, "published_avg4"].iloc[0])
            gc = float(s.loc[s.model == mi, "corrected_avg3"].iloc[0]
                       - s.loc[s.model == mj, "corrected_avg3"].iloc[0])
            pairs.append({"pair": f"{mi} vs {mj}", "gap_published": round(gp, 4),
                          "gap_corrected": round(gc, 4),
                          "sign_flip": bool(np.sign(gp) != np.sign(gc)),
                          "pub_below_sigma": abs(gp) < SIGMA_RUN,
                          "cor_below_sigma": abs(gc) < SIGMA_RUN})
    p = pd.DataFrame(pairs)
    print(p.to_string(index=False))
    p.to_csv(OUT / "pairwise_gaps.csv", index=False)
    print(f"\n  sign flips: {int(p.sign_flip.sum())} of {len(p)}")
    print(f"  comparisons below the noise floor: published {int(p.pub_below_sigma.sum())}, "
          f"corrected {int(p.cor_below_sigma.sum())}")

    # --- how much was CVC-300 double-counting worth on its own? ---
    print("\n=== contribution of CVC-300 double-counting alone ===")
    rows = []
    for model in sorted(df.model.unique()):
        with300 = float(np.mean([pub.loc[model, d] for d in EXTERNAL_PUBLISHED]))
        no300 = float(np.mean([pub.loc[model, d] for d in EXTERNAL_CORRECTED]))
        rows.append({"model": model, "avg_with_cvc300": round(with300, 4),
                     "avg_without_cvc300": round(no300, 4),
                     "delta": round(no300 - with300, 4),
                     "cvc300_mean": round(pub.loc[model, "cvc-300"], 4),
                     "colondb_mean": round(pub.loc[model, "cvc-colondb"], 4)})
    c3 = pd.DataFrame(rows)
    print(c3.to_string(index=False))
    c3.to_csv(OUT / "cvc300_effect.csv", index=False)

    # Independent confirmation of identity: CVC-300 must score exactly the same
    # as CVC-ColonDB frames 149-208, which is where its images live.
    import re as _re
    cd = df[df.dataset == "cvc-colondb"].copy()
    c3rows = df[df.dataset == "cvc-300"].copy()
    cd["frame"] = [int(_re.sub(r"\D", "", str(x))) for x in cd["name"]]
    blk = cd[(cd.frame >= 149) & (cd.frame <= 208)]
    rest = cd[(cd.frame < 149) | (cd.frame > 208)]
    diff = (c3rows.groupby("model")[METRIC].mean()
            - blk.groupby("model")[METRIC].mean()).abs().mean()
    print(f"\n  CVC-300 vs CVC-ColonDB frames 149-208: mean |difference| = {diff:.6f}")
    print(f"  -- identical scores confirm they are the same images")
    print(f"  the duplicated block scores {blk[METRIC].mean():.4f} against "
          f"{rest[METRIC].mean():.4f} for the rest of CVC-ColonDB "
          f"({blk[METRIC].mean() - rest[METRIC].mean():+.4f})")
    print(f"  -- so reporting it separately double-counts an unusually easy block")
    pd.DataFrame({
        "model": sorted(df.model.unique()),
        "cvc300": [c3rows[c3rows.model == m][METRIC].mean() for m in sorted(df.model.unique())],
        "colondb_149_208": [blk[blk.model == m][METRIC].mean() for m in sorted(df.model.unique())],
        "colondb_rest": [rest[rest.model == m][METRIC].mean() for m in sorted(df.model.unique())],
    }).to_csv(OUT / "cvc300_identity_check.csv", index=False)
    print(f"\nartifacts in {OUT}")


if __name__ == "__main__":
    main()
