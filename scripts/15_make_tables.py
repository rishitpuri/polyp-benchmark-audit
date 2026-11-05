"""Generate the manuscript tables from the result CSVs.

Written to paper/tables/ and pulled into main.tex via \\input, on the same
principle as the macros: no table body is typed by hand, so a table cannot
disagree with the experiment behind it.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "tables"
SIGMA_RUN = 0.0053

PRETTY_DS = {
    "kvasir-seg": "Kvasir-SEG", "kvasir-sessile": "Kvasir-Sessile",
    "cvc-clinicdb": "CVC-ClinicDB", "cvc-colondb": "CVC-ColonDB",
    "cvc-300": "CVC-300", "etis-larib": "ETIS-Larib",
}
PRETTY_MODEL = {
    "unet": "U-Net", "unetpp": "U-Net++", "segformer": "SegFormer-B0",
    "segformer_b2": "SegFormer-B2", "unet_pvt": "U-Net+PVTv2",
}
ORDER = ["kvasir-seg", "kvasir-sessile", "cvc-clinicdb", "cvc-colondb",
         "cvc-300", "etis-larib"]
# containment relations established in the Results
CONTAINED_IN = {"cvc-300": "CVC-ColonDB", "kvasir-sessile": "Kvasir-SEG"}


def table1():
    """Per-dataset audit summary."""
    idx = pd.read_csv(ROOT / "outputs/contamination/image_index.csv")
    red = pd.read_csv(ROOT / "outputs/contamination/within_dataset_redundancy.csv").set_index("dataset")

    lines = [
        r"\begin{tabular}{lrrrrrl}", r"\toprule",
        r"Dataset & Files & Unique & Exact & \multicolumn{2}{c}{Near-duplicate (\%)} & Contained in \\",
        r"\cmidrule(lr){5-6}",
        r" & & & dupl. & $r>0.95$ & $r>0.90$ & \\",
        r"\midrule",
    ]
    for ds in ORDER:
        g = idx[idx.dataset == ds]
        n, uniq = len(g), g.pixel_md5.nunique()
        lines.append(
            f"{PRETTY_DS[ds]} & {n} & {uniq} & {n-uniq} & "
            f"{100*red.loc[ds,'frac_nn_r>0.95']:.1f} & "
            f"{100*red.loc[ds,'frac_nn_r>0.90']:.1f} & "
            f"{CONTAINED_IN.get(ds, '--')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table1_audit.tex").write_text("\n".join(lines) + "\n")
    return "table1_audit.tex"


def table2():
    """Every measured consequence, in Dice and in noise-floor units."""
    shift = pd.read_csv(ROOT / "outputs/rescore/external_average_shift.csv")
    cdb = pd.read_csv(ROOT / "outputs/inflation/inflation_pooled_clinicdb.csv")
    kv = pd.read_csv(ROOT / "outputs/inflation/inflation_pooled_kvasir.csv")
    cdb_i = cdb[cdb.dataset.str.contains("in-domain")].iloc[0]
    kv_i = kv[kv.dataset.str.contains("in-domain")].iloc[0]
    prior = pd.read_csv(ROOT / "outputs/prior/prior_summary.csv").set_index("dataset")
    pn_path = ROOT / "outputs/pranet/pranet_summary.csv"
    pn = pd.read_csv(pn_path).iloc[0] if pn_path.exists() else None

    rows = []
    rows.append((
        "Dropping CVC-300 from the suite average (5 architectures)",
        f"{shift.delta.mean():+.4f}",
        f"{abs(shift.delta.mean())/SIGMA_RUN:.1f}",
        f"top rank reverses; {int(shift.rank_change.ne(0).sum())}/{len(shift)} move",
    ))
    if pn is not None:
        rows.append((
            "Dropping CVC-300, released PraNet checkpoint",
            f"{pn['shift']:+.4f}", f"{abs(pn['shift'])/SIGMA_RUN:.1f}",
            "three unseen datasets become two",
        ))
    rows.append((
        "Random vs sequence-disjoint splits, CVC-ClinicDB",
        f"{cdb_i.inflation:+.4f}", f"{cdb_i.in_sigma_run:.1f}",
        # npj requires the actual p-value rather than a bound; the exact value
        # underflows here, so the order of magnitude from the normal
        # approximation is quoted (see Statistics and reproducibility)
        f"{int(cdb_i.n_images)//5} frames $\\times$ 5 arch., paired, "
        f"$p \\approx 10^{{{cdb_i.wilcoxon_log10_p:.0f}}}$",
    ))
    rows.append((
        "\\quad adjusted for training diversity",
        f"{cdb_i.inflation - cdb[~cdb.dataset.str.contains('in-domain')].inflation.mean():+.4f}",
        f"{(cdb_i.inflation - cdb[~cdb.dataset.str.contains('in-domain')].inflation.mean())/SIGMA_RUN:.1f}",
        "estimate, see text",
    ))
    rows.append((
        "Random vs grouped splits, Kvasir-SEG",
        f"{kv_i.inflation:+.4f}", f"{kv_i.in_sigma_run:.1f}",
        f"3/5 intervals cross zero; $p \\approx 10^{{{kv_i.wilcoxon_log10_p:.0f}}}$",
    ))
    rows.append((
        "Content-free position prior, Kvasir-SEG",
        f"{prior.loc['kvasir-seg','prior_only_dice']:.4f}", "--",
        f"vs {prior.loc['kvasir-seg','trained_dice']:.4f} trained",
    ))
    rows.append((
        "Content-free position prior, CVC-ClinicDB",
        f"{prior.loc['cvc-clinicdb','prior_only_dice']:.4f}", "--",
        f"vs {prior.loc['cvc-clinicdb','trained_dice']:.4f} trained",
    ))

    # tabularx with two flexible columns: the effect description and the note
    # are both variable-length, and a plain tabular overruns the text block.
    lines = [
        r"\begin{tabularx}{\textwidth}{@{}Xrr>{\raggedright\arraybackslash}p{3.6cm}@{}}",
        r"\toprule",
        r"Effect & Dice & $/\sigma_{\mathrm{run}}$ & Note \\", r"\midrule",
    ]
    for desc, dice, sig, note in rows:
        lines.append(f"{desc} & {dice} & {sig} & {note} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    (OUT / "table2_effects.tex").write_text("\n".join(lines) + "\n")
    return "table2_effects.tex"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (table1, table2):
        name = fn()
        print(f"wrote {name}")
        print((OUT / name).read_text())


if __name__ == "__main__":
    main()
