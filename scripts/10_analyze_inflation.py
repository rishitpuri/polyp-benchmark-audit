"""Leakage inflation: random frame splits vs sequence-disjoint splits.

A naive comparison of the two arms' reported Dice would be confounded. The
arms necessarily use different test folds -- that is what distinguishes the
protocols -- so a difference in mean Dice mixes the effect of leakage with
the effect of the test sets containing different images.

Both arms are five-fold partitions of all 612 CVC-ClinicDB frames, so each
frame is in the test fold exactly once per arm. Pooling the five folds
therefore yields one held-out prediction for every frame under each protocol,
on identical images. The comparison is then exactly paired, and the per-image
difference isolates leakage: under the random protocol a frame is scored by a
model trained on its neighbours from the same sequence; under the
sequence-disjoint protocol it is scored by a model that never saw that
sequence.

External sets are scored by all five fold models in each arm, so those are
averaged per image before pairing.

Everything is read against sigma_run = 0.0053, the run-to-run noise floor
measured in paper 1 under this same recipe.
"""

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "outputs" / "paper2"
OUT = ROOT / "outputs" / "inflation"

SIGMA_RUN = 0.0053
METRIC = "dice"
N_BOOT = 10000

# experiment -> (leaky arm, leak-free arm, in-domain dataset label, externals)
EXPERIMENTS = {
    "clinicdb": (("clinicdb_random", "clinicdb_seqdisjoint"), "cvc-clinicdb",
                 [("kvasir-seg-ext", "Kvasir-SEG (external)"),
                  ("cvc-colondb", "CVC-ColonDB (external)"),
                  ("etis-larib", "ETIS-Larib (external)")]),
    "kvasir": (("kvasir_imagelevel", "kvasir_grouped_tau0.90"), "kvasir-seg",
               [("cvc-clinicdb-ext", "CVC-ClinicDB (external)"),
                ("cvc-colondb", "CVC-ColonDB (external)"),
                ("etis-larib", "ETIS-Larib (external)")]),
}
PATTERN = re.compile(
    r"perimage_(?P<model>.+?)_seed(?P<seed>\d+)_"
    r"(?P<arm>clinicdb_random|clinicdb_seqdisjoint|kvasir_imagelevel|kvasir_grouped_tau0\.90)"
    r"_fold(?P<fold>\d)\.csv$")

ARMS = ()  # bound per experiment in main()
INDOMAIN = ""


def load():
    rows = []
    for path in sorted(glob.glob(str(IN / "perimage_*.csv"))):
        m = PATTERN.search(Path(path).name)
        if not m:
            continue
        d = pd.read_csv(path)
        d["model"] = m.group("model")
        d["arm"] = m.group("arm")
        d["fold"] = int(m.group("fold"))
        rows.append(d)
    if not rows:
        raise SystemExit(f"no parsable per-image CSVs in {IN}")
    return pd.concat(rows, ignore_index=True)


def boot_ci(x, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_table(df, dataset, label):
    """Pair per-image scores between the two arms on identical images."""
    sub = df[df.dataset == dataset]
    # in-domain: each image appears once per (arm, model); external: five times
    per = sub.groupby(["arm", "model", "name"])[METRIC].mean().reset_index()
    wide = per.pivot_table(index=["model", "name"], columns="arm", values=METRIC)
    if not set(ARMS).issubset(wide.columns):
        return None, None  # the second arm has not produced results yet
    wide = wide.dropna(subset=list(ARMS))
    if wide.empty:
        return None, None

    rows = []
    for model, g in wide.groupby("model"):
        d = (g[ARMS[0]] - g[ARMS[1]]).values  # random minus sequence-disjoint
        lo, hi = boot_ci(d)
        try:
            p = wilcoxon(g[ARMS[0]].values, g[ARMS[1]].values).pvalue
        except ValueError:
            p = float("nan")
        rows.append({
            "dataset": label, "model": model, "n_images": len(d),
            "random": round(float(g[ARMS[0]].mean()), 4),
            "seq_disjoint": round(float(g[ARMS[1]].mean()), 4),
            "inflation": round(float(d.mean()), 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "in_sigma_run": round(float(d.mean()) / SIGMA_RUN, 2),
            "wilcoxon_p": f"{p:.2e}" if p == p else "nan",
        })

    pooled_d = (wide[ARMS[0]] - wide[ARMS[1]]).values
    lo, hi = boot_ci(pooled_d)
    pooled = {
        "dataset": label, "model": "POOLED", "n_images": len(pooled_d),
        "random": round(float(wide[ARMS[0]].mean()), 4),
        "seq_disjoint": round(float(wide[ARMS[1]].mean()), 4),
        "inflation": round(float(pooled_d.mean()), 4),
        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
        "in_sigma_run": round(float(pooled_d.mean()) / SIGMA_RUN, 2),
        "wilcoxon_p": f"{wilcoxon(wide[ARMS[0]].values, wide[ARMS[1]].values).pvalue:.2e}",
    }
    return pd.DataFrame(rows), pooled


def run_experiment(df, key):
    global ARMS, INDOMAIN
    arms, indomain, externals = EXPERIMENTS[key]
    ARMS, INDOMAIN = arms, indomain
    df = df[df.arm.isin(arms)]
    if df.empty:
        print(f"\n### {key}: no results yet, skipping")
        return

    print(f"\n{'='*78}\n### experiment: {key}   ({arms[0]} vs {arms[1]})\n{'='*78}")
    have = df.groupby(["arm", "model"])["fold"].nunique().unstack(fill_value=0)
    print("=== folds available per arm and architecture (need 5 each) ===")
    print(have.to_string())
    complete = (have == 5).all().all() if set(ARMS).issubset(have.index) else False
    print(f"\n  complete: {complete}"
          + ("" if complete else "   -- numbers below are PRELIMINARY"))

    print(f"\n=== coverage check: every frame held out exactly once per arm ===")
    ind = df[df.dataset == INDOMAIN]
    for arm in ARMS:
        a = ind[ind.arm == arm]
        if a.empty:
            continue
        counts = a.groupby(["model", "name"]).size()
        print(f"  {arm:<22} images {a.name.nunique():>4}  "
              f"appearances per (model,image): min {counts.min()} max {counts.max()}")

    indomain_label = f"{INDOMAIN} (in-domain)"
    all_rows, pooled_rows = [], []
    for ds, label in [(INDOMAIN, indomain_label)] + externals:
        tbl, pooled = paired_table(df, ds, label)
        if tbl is None:
            continue
        all_rows.append(tbl)
        pooled_rows.append(pooled)

    if not all_rows:
        print("  no paired data yet")
        return

    full = pd.concat(all_rows, ignore_index=True)
    pooled = pd.DataFrame(pooled_rows)

    print(f"\n=== inflation ({arms[0]} minus {arms[1]}) ===")
    for label in full.dataset.unique():
        print(f"\n--- {label}")
        t = full[full.dataset == label].drop(columns="dataset")
        print(t.to_string(index=False))
        p = pooled[pooled.dataset == label].iloc[0]
        print(f"  POOLED  inflation {p['inflation']:+.4f} Dice "
              f"[{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}]  "
              f"= {p['in_sigma_run']:+.1f} x sigma_run   p={p['wilcoxon_p']}")

    full.to_csv(OUT / f"inflation_by_model_{key}.csv", index=False)
    pooled.to_csv(OUT / f"inflation_pooled_{key}.csv", index=False)

    # does the leakage change which architecture looks best in domain?
    ind_tbl = full[full.dataset == indomain_label]
    if not ind_tbl.empty:
        print("\n=== ranking under each protocol (in-domain) ===")
        r = ind_tbl[["model", "random", "seq_disjoint"]].copy()
        r["rank_random"] = r["random"].rank(ascending=False).astype(int)
        r["rank_seq"] = r["seq_disjoint"].rank(ascending=False).astype(int)
        r["moved"] = r.rank_random - r.rank_seq
        print(r.sort_values("rank_random").to_string(index=False))
        flips = int((r.moved != 0).sum())
        print(f"\n  architectures changing rank: {flips} of {len(r)}")
        r.to_csv(OUT / f"ranking_shift_{key}.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    for key in EXPERIMENTS:
        run_experiment(df, key)
    print(f"\nartifacts in {OUT}")


if __name__ == "__main__":
    main()
