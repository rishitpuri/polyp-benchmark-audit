"""Split files for the two GPU experiments.

Experiment 1 -- CVC-ClinicDB, random vs sequence-disjoint.
  The same architectures are trained twice over five folds: once with frames
  assigned to folds at random, as the literature does, and once with whole
  source sequences assigned to folds, using the official 29-sequence table.
  The difference is the inflation attributable to leakage. Train/val/test
  sizes are matched as closely as whole-sequence assignment allows, so the
  comparison is not confounded by training-set size.

Experiment 2 -- Kvasir-SEG, image-level vs group-disjoint.
  Paper 1 cross-validated at image level and listed the resulting leakage as
  a limitation. Kvasir-SEG has no order and no ground-truth grouping, so
  leak-freedom cannot be certified; folds are emitted at several grouping
  strictnesses and the downstream estimate is reported across them.

Every CSV carries `name`, `image_file`, `mask_file`, `fold`, `split` so that
the adapter can resolve files regardless of extension.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLITS = ROOT / "outputs" / "splits"
OUT = SPLITS / "training"

N_FOLDS = 5
VAL_FRAC = 0.125  # of the non-test remainder, matching paper 1's 700/100/200


def assign_groups_to_folds(group_sizes, n_folds):
    """Greedy largest-first balancing of whole groups across folds."""
    loads = np.zeros(n_folds, dtype=int)
    assign = {}
    for g in group_sizes.sort_values(ascending=False).index:
        f = int(np.argmin(loads))
        assign[g] = f
        loads[f] += int(group_sizes[g])
    return assign, loads


def build_fold_table(df, group_col, n_folds, seed=0):
    """For each fold: that fold is test, one adjacent fold supplies val."""
    sizes = df.groupby(group_col).size()
    assign, loads = assign_groups_to_folds(sizes, n_folds)
    df = df.copy()
    df["fold"] = df[group_col].map(assign)

    tables = {}
    rng = np.random.default_rng(seed)
    for k in range(n_folds):
        t = df.copy()
        t["split"] = "train"
        t.loc[t.fold == k, "split"] = "test"
        # carve validation out of the remaining groups, whole groups only
        rest = sizes.index[[assign[g] != k for g in sizes.index]]
        order = rng.permutation(list(rest))
        target = int(round(VAL_FRAC * int(sizes[list(rest)].sum())))
        chosen, got = [], 0
        for g in order:
            if got >= target:
                break
            chosen.append(g)
            got += int(sizes[g])
        t.loc[t[group_col].isin(chosen), "split"] = "val"
        tables[k] = t
    return tables, loads


def write(tables, prefix, cols):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, t in tables.items():
        path = OUT / f"{prefix}_fold{k}.csv"
        t[cols].to_csv(path, index=False)
        c = t.split.value_counts()
        rows.append({"split_file": path.name, "train": int(c.get("train", 0)),
                     "val": int(c.get("val", 0)), "test": int(c.get("test", 0))})
    return pd.DataFrame(rows)


def main():
    cols = ["name", "image_file", "mask_file", "group", "fold", "split"]

    # ---------------- CVC-ClinicDB ----------------
    cdb = pd.read_csv(SPLITS / "cvc-clinicdb_groups_folds.csv")
    cdb["name"] = [Path(x).stem for x in cdb["image"]]
    cdb["image_file"] = cdb["image"]
    cdb["mask_file"] = cdb["image"]
    cdb["frame"] = [int(re.sub(r"\D", "", s)) for s in cdb["image"]]

    print("=== CVC-ClinicDB, sequence-disjoint (29 official sequences) ===")
    seq_tables, seq_loads = build_fold_table(cdb, "group", N_FOLDS)
    seq_summary = write(seq_tables, "clinicdb_seqdisjoint", cols)
    print(seq_summary.to_string(index=False))

    print("\n=== CVC-ClinicDB, random frame split (the published protocol) ===")
    rnd = cdb.copy()
    rng = np.random.default_rng(0)
    rnd["group"] = rng.permutation(len(rnd))  # every frame its own group
    rnd_tables, _ = build_fold_table(rnd, "group", N_FOLDS)
    rnd_summary = write(rnd_tables, "clinicdb_random", cols)
    print(rnd_summary.to_string(index=False))

    # how much do the two protocols differ in what they leak?
    print("\n=== leakage check on the emitted folds ===")
    for tag, tables in (("sequence-disjoint", seq_tables), ("random", rnd_tables)):
        shared = []
        for k, t in tables.items():
            tr = set(t.loc[t.split == "train", "group"]) if tag == "sequence-disjoint" else set(
                cdb.loc[cdb["name"].isin(t.loc[t.split == "train", "name"]), "group"])
            te = set(t.loc[t.split == "test", "group"]) if tag == "sequence-disjoint" else set(
                cdb.loc[cdb["name"].isin(t.loc[t.split == "test", "name"]), "group"])
            shared.append(len(tr & te) / max(1, len(te)))
        print(f"  {tag:<18} mean fraction of test sequences also in train: {np.mean(shared):.3f}")

    # ---------------- Kvasir-SEG ----------------
    # Only two levels are informative: tau=0.90 actually groups (882 groups
    # for 1000 images), while tau>=0.93 is within a few images of no grouping
    # at all. The image-level arm is generated here rather than reused from
    # paper 1 so that both arms share an identical fold construction and the
    # only difference is whether near-duplicates are kept together.
    print("\n=== Kvasir-SEG: image-level vs grouped (matched construction) ===")
    rows = []
    kv_base = pd.read_csv(SPLITS / "kvasir-seg_groups_folds_tau0.90.csv")
    kv_base["name"] = [Path(x).stem for x in kv_base["image"]]
    kv_base["image_file"] = kv_base["image"]
    kv_base["mask_file"] = kv_base["image"]

    imglevel = kv_base.copy()
    rng2 = np.random.default_rng(0)
    imglevel["group"] = rng2.permutation(len(imglevel))
    for tag, frame in (("kvasir_imagelevel", imglevel), ("kvasir_grouped_tau0.90", kv_base)):
        tables, _ = build_fold_table(frame, "group", N_FOLDS)
        s = write(tables, tag, cols)
        rows.append({"arm": tag, "n_groups": int(frame.group.nunique()),
                     "mean_train": int(s.train.mean()), "mean_test": int(s.test.mean())})
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\nsplit files written to {OUT}")
    print(f"total: {len(list(OUT.glob('*.csv')))} CSVs")


if __name__ == "__main__":
    main()
