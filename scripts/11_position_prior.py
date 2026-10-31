"""How much benchmark Dice is recoverable without looking at the image?

Benchmark polyps are usually centred and well framed, which the SUN-SEG paper
documented as a centre-bias statistic. The question here is what that bias is
worth in the metric the field reports.

The predictor sees no image content at all. It averages the training-fold
masks into a single probability map, thresholds it at the value that maximises
training Dice, and emits that one fixed shape for every test image. Any Dice
it achieves is attributable purely to the position and size prior of the
dataset, not to recognising a lesion.

Folds are the leak-free ones built in script 06/09 where available, so the
prior is never estimated on an image it is scored against. CVC-ColonDB and
ETIS-Larib are zero-shot test sets with no training protocol, so a five-fold
partition is constructed here solely to keep the prior honest.

Reported against the trained-model Dice from the leak-free arms of the GPU
experiments, so the comparison is prior vs genuinely-generalising model.
"""

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLITS = ROOT / "outputs" / "splits"
RESULTS = ROOT / "outputs" / "paper2"
OUT = ROOT / "outputs" / "prior"

GRID = 352
N_FOLDS = 5
THRESHOLDS = np.linspace(0.05, 0.95, 37)

DATASETS = {
    "kvasir-seg": DATA / "Kvasir-SEG",
    "cvc-clinicdb": DATA / "external/cvc-clinicdb",
    "cvc-colondb": DATA / "external/cvc-colondb",
    "etis-larib": DATA / "external/etis-larib",
}

# leak-free trained-model reference: which arm supplies each dataset's Dice
REFERENCE_ARM = {
    "kvasir-seg": "kvasir_grouped_tau0.90",
    "cvc-clinicdb": "clinicdb_seqdisjoint",
    "cvc-colondb": "clinicdb_seqdisjoint",
    "etis-larib": "clinicdb_seqdisjoint",
}


def load_masks(root):
    mask_dir = root / "masks"
    out = {}
    for p in sorted(mask_dir.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            continue
        with Image.open(p) as im:
            g = np.asarray(im.convert("L").resize((GRID, GRID), Image.NEAREST))
        out[p.stem] = (g > 127).astype(np.float32)
    return out


def dice(pred, gt):
    s = pred.sum() + gt.sum()
    if s == 0:
        return 1.0
    return float(2.0 * (pred * gt).sum() / s)


# Only these two datasets have a defensible grouping: CVC-ClinicDB from its
# official frame-range table, Kvasir-SEG from the tau=0.90 similarity groups.
# CVC-ColonDB, ETIS-Larib and CVC-300 are zero-shot test sets for which no
# grouping could be certified, so a plain partition is constructed instead.
GROUPED_FOLDS = {
    "cvc-clinicdb": "cvc-clinicdb_groups_folds.csv",
    "kvasir-seg": "kvasir-seg_groups_folds_tau0.90.csv",
}


def folds_for(name, stems):
    """Leak-free folds where we have them; otherwise a plain partition."""
    candidates = [SPLITS / GROUPED_FOLDS[name]] if name in GROUPED_FOLDS else []
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df["stem"] = [Path(x).stem for x in df["image"]]
            df = df[df.stem.isin(stems)]
            if len(df) == len(stems):
                return dict(zip(df.stem, df.fold)), f"leak-free ({path.name})"
    rng = np.random.default_rng(0)
    order = rng.permutation(sorted(stems))
    return ({s: i % N_FOLDS for i, s in enumerate(order)},
            "constructed partition (test-only dataset)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # trained-model reference Dice from the leak-free arms
    ref = {}
    for path in glob.glob(str(RESULTS / "perimage_*.csv")):
        m = re.search(r"perimage_(.+?)_seed\d+_(clinicdb_\w+|kvasir_\w+?[\d.]*)_fold\d\.csv$",
                      Path(path).name)
        if not m:
            continue
        arm = m.group(2)
        d = pd.read_csv(path)
        for ds, g in d.groupby("dataset"):
            ref.setdefault((ds.replace("-ext", ""), arm), []).extend(g["dice"].tolist())

    rows = []
    for name, root in DATASETS.items():
        masks = load_masks(root)
        assign, source = folds_for(name, set(masks))
        print(f"\n=== {name}: {len(masks)} masks, folds = {source}")

        per_image = []
        for k in range(N_FOLDS):
            train = [s for s in masks if assign[s] != k]
            test = [s for s in masks if assign[s] == k]
            if not train or not test:
                continue
            prior = np.mean([masks[s] for s in train], axis=0)
            # threshold chosen on the training fold only
            best_t, best_d = 0.5, -1.0
            for t in THRESHOLDS:
                pred = (prior > t).astype(np.float32)
                d = np.mean([dice(pred, masks[s]) for s in train])
                if d > best_d:
                    best_t, best_d = float(t), float(d)
            pred = (prior > best_t).astype(np.float32)
            for s in test:
                per_image.append({"image": s, "fold": k, "threshold": best_t,
                                  "dice": dice(pred, masks[s])})

        pi = pd.DataFrame(per_image)
        pi.to_csv(OUT / f"prior_perimage_{name}.csv", index=False)

        arm = REFERENCE_ARM[name]
        trained = ref.get((name, arm))
        trained_mean = float(np.mean(trained)) if trained else float("nan")
        prior_mean = float(pi["dice"].mean())
        rows.append({
            "dataset": name, "n": len(pi),
            "prior_only_dice": round(prior_mean, 4),
            "trained_dice": round(trained_mean, 4),
            "pct_of_trained": round(100 * prior_mean / trained_mean, 1) if trained else None,
            "mean_area_frac": round(float(np.mean([m.mean() for m in masks.values()])), 4),
            "fold_source": source,
        })
        print(f"  content-free prior Dice {prior_mean:.4f}   "
              f"trained ({arm}) {trained_mean:.4f}   "
              f"= {100*prior_mean/trained_mean:.1f}% of trained" if trained
              else f"  content-free prior Dice {prior_mean:.4f}")

    res = pd.DataFrame(rows)
    print("\n=== what a predictor that never sees the image achieves ===")
    print(res.to_string(index=False))
    res.to_csv(OUT / "prior_summary.csv", index=False)
    print(f"\nartifacts in {OUT}")


if __name__ == "__main__":
    main()
