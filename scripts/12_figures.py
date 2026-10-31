"""Manuscript figures.

Every panel is generated from the result CSVs, so no number in a figure can
drift from the experiment that produced it -- the same discipline paper 1
used for its tables.

  fig1  what the benchmark suite actually contains (contamination)
  fig2  consequence of counting CVC-300 as an independent test set
  fig3  leakage inflation, read against the run-to-run noise floor
  fig4  how much Dice a predictor gets without seeing the image
  fig5  recoverability of source-video structure
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "xtick.labelsize": 7, "ytick.labelsize": 7,
})

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "figures"
SIGMA_RUN = 0.0053

PRETTY = {"unet": "U-Net", "unetpp": "U-Net++", "segformer": "SegFormer-B0",
          "segformer_b2": "SegFormer-B2", "unet_pvt": "U-Net+PVTv2"}
DS_PRETTY = {"kvasir-seg": "Kvasir-SEG", "kvasir-sessile": "Kvasir-Sessile",
             "cvc-clinicdb": "CVC-ClinicDB", "cvc-colondb": "CVC-ColonDB",
             "cvc-300": "CVC-300", "etis-larib": "ETIS-Larib"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def fig1_contamination():
    summary = pd.read_csv(ROOT / "outputs/contamination/cross_dataset_summary.csv")
    red = pd.read_csv(ROOT / "outputs/contamination/within_dataset_redundancy.csv")
    order = ["kvasir-seg", "kvasir-sessile", "cvc-clinicdb", "cvc-colondb",
             "cvc-300", "etis-larib"]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1),
                             gridspec_kw={"width_ratios": [1.15, 0.95, 0.9]})
    fig.subplots_adjust(wspace=0.62)  # leave room for (a)'s colorbar

    # The match criterion is the one stated in Methods: both the correlation
    # and the perceptual-hash signal must agree.
    M = np.full((len(order), len(order)), np.nan)
    for _, r in summary.iterrows():
        i, j = order.index(r["query"]), order.index(r["reference"])
        M[i, j] = 100.0 * r["n_both"] / r["n_query"]
    np.fill_diagonal(M, np.nan)
    ax = axes[0]
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=100)
    ax.set_xticks(range(len(order)), [DS_PRETTY[o] for o in order], rotation=45, ha="right")
    ax.set_yticks(range(len(order)), [DS_PRETTY[o] for o in order])
    for i in range(len(order)):
        for j in range(len(order)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "\u2013", ha="center", va="center", color="0.6")
            else:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                        color="white" if M[i, j] > 50 else "black", fontsize=7)
    ax.set_title("(a) % of images matched to a twin\nin the other dataset")
    ax.set_xlabel("reference dataset")
    ax.set_ylabel("query dataset")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cb.set_label("%", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    ax = axes[1]
    red = red.set_index("dataset").loc[order]
    x = np.arange(len(order))
    for frac, lab, c in [("frac_nn_r>0.90", "r > 0.90", "#c6dbef"),
                         ("frac_nn_r>0.95", "r > 0.95", "#6baed6"),
                         ("frac_nn_r>0.98", "r > 0.98", "#2171b5")]:
        ax.bar(x, 100 * red[frac].values, color=c, label=lab, width=0.62)
    ax.set_xticks(x, [DS_PRETTY[o] for o in order], rotation=45, ha="right")
    ax.set_ylabel("% of images")
    ax.set_title("(b) images with a near-duplicate\nwithin the same dataset")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(0, 100)

    # (c) the two containments are detected very differently depending on the
    # signal used: exact pixel hashing finds Kvasir-Sessile not at all.
    ax = axes[2]
    cases = [("cvc-300", "cvc-colondb", "CVC-300\nin CVC-ColonDB"),
             ("kvasir-sessile", "kvasir-seg", "Kvasir-Sessile\nin Kvasir-SEG")]
    sigs = [("pixel MD5", "#d62728"), ("phash $=0$", "#ff9896"),
            ("phash $\\leq 5$", "#9ecae1"), ("corr $>0.999$", "#2171b5")]
    idxmeta = pd.read_csv(ROOT / "outputs/contamination/image_index.csv")
    w, xs = 0.2, np.arange(len(cases))
    for si, (lab, colour) in enumerate(sigs):
        vals = []
        for q, ref, _ in cases:
            r = summary[(summary["query"] == q) & (summary.reference == ref)].iloc[0]
            n = r["n_query"]
            if lab == "pixel MD5":
                qh = set(idxmeta[idxmeta.dataset == ref].pixel_md5)
                v = 100.0 * idxmeta[idxmeta.dataset == q].pixel_md5.isin(qh).sum() / n
            elif lab == "phash $=0$":
                v = 100.0 * r["n_hamming<=0"] / n
            elif lab == "phash $\\leq 5$":
                v = 100.0 * r["n_hamming<=5"] / n
            else:
                v = 100.0 * r["n_r>0.999"] / n
            vals.append(v)
        pos = xs + (si - 1.5) * w
        ax.bar(pos, vals, width=w, label=lab, color=colour)
        # a zero-height bar is indistinguishable from a missing one, and the
        # zero is the point: exact hashing detects none of Kvasir-Sessile
        for px, v in zip(pos, vals):
            ax.text(px, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=6,
                    color="#d62728" if v == 0 else "0.25",
                    fontweight="bold" if v == 0 else "normal")
    ax.set_xticks(xs, [c[2] for c in cases], fontsize=7)
    ax.set_ylabel("% of the contained set detected")
    ax.set_title("(c) detection depends on the signal")
    ax.set_ylim(0, 142)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(fontsize=6.5, loc="upper center", ncol=2, columnspacing=0.8,
              handlelength=1.2, borderpad=0.3)
    save(fig, "fig1_contamination")


def fig2_cvc300():
    s = pd.read_csv(ROOT / "outputs/rescore/external_average_shift.csv")
    s["label"] = s.model.map(PRETTY).fillna(s.model)
    s = s.sort_values("published_avg4", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    for _, r in s.iterrows():
        moved = r.rank_published != r.rank_corrected
        ax.plot([0, 1], [r.published_avg4, r.corrected_avg3], "-o", ms=4,
                lw=2.0 if moved else 1.0,
                color="#d62728" if moved else "0.55",
                zorder=3 if moved else 2)
        ax.annotate(r.label, (0, r.published_avg4), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=7,
                    color="#d62728" if moved else "0.3")
    ax.set_xlim(-0.75, 1.35)
    ax.set_xticks([0, 1], ["published\n(4 datasets)", "corrected\n(3 datasets)"])
    ax.set_ylabel("mean external Dice")
    ax.set_title("(a) removing CVC-300 reverses\nthe top-ranked architecture")

    ax = axes[1]
    x = np.arange(len(s))
    ax.bar(x, s.delta.abs() / SIGMA_RUN, color="#4c78a8", width=0.6)
    ax.axhline(1.0, color="#d62728", ls="--", lw=1,
               label=r"$\sigma_{\mathrm{run}}$ = 0.0053")
    ax.set_xticks(x, s.label, rotation=45, ha="right")
    ax.set_ylabel(r"|shift| / $\sigma_{\mathrm{run}}$")
    ax.set_title("(b) every architecture shifts by\nseveral times the noise floor")
    ax.legend(fontsize=7)
    save(fig, "fig2_cvc300_double_counting")


def fig3_inflation():
    c = pd.read_csv(ROOT / "outputs/inflation/inflation_by_model_clinicdb.csv")
    k = pd.read_csv(ROOT / "outputs/inflation/inflation_by_model_kvasir.csv")
    cp = pd.read_csv(ROOT / "outputs/inflation/inflation_pooled_clinicdb.csv")
    kp = pd.read_csv(ROOT / "outputs/inflation/inflation_pooled_kvasir.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), sharey=False)
    # the two panels carry independent y-tick labels; without generous spacing
    # panel (b)'s labels sit on top of panel (a)'s data points
    fig.subplots_adjust(wspace=0.42)
    for ax, tbl, pooled, title, nd in (
            (axes[0], c, cp, "(a) CVC-ClinicDB\n90% near-duplicate rate", "cvc-clinicdb"),
            (axes[1], k, kp, "(b) Kvasir-SEG\n15% near-duplicate rate", "kvasir-seg")):
        ind = tbl[tbl.dataset.str.contains(r"in-domain")].copy()
        ind["label"] = ind.model.map(PRETTY).fillna(ind.model)
        ind = ind.sort_values("inflation")
        y = np.arange(len(ind))
        err = np.vstack([ind.inflation - ind.ci_lo, ind.ci_hi - ind.inflation])
        ax.errorbar(ind.inflation, y, xerr=err, fmt="o", ms=4, lw=1.2,
                    color="#4c78a8", capsize=2.5)
        p = pooled[pooled.dataset.str.contains("in-domain")].iloc[0]
        ax.axvline(p.inflation, color="#d62728", lw=1.2,
                   label=f"pooled {p.inflation:+.4f} ({p.in_sigma_run:.1f}$\\sigma$)")
        ax.axvspan(-SIGMA_RUN, SIGMA_RUN, color="0.85", zorder=0,
                   label=r"$\pm\sigma_{\mathrm{run}}$")
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_yticks(y, ind.label)
        ax.set_xlabel("Dice inflation from leaky splits")
        ax.set_title(title)
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9, frameon=True,
                  edgecolor="none")
        ax.set_ylim(-0.8, len(ind) - 0.2)
    save(fig, "fig3_leakage_inflation")


def fig4_prior():
    p = pd.read_csv(ROOT / "outputs/prior/prior_summary.csv")
    p["label"] = p.dataset.map(DS_PRETTY).fillna(p.dataset)
    p = p.sort_values("prior_only_dice", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    x = np.arange(len(p))
    ax.bar(x - 0.19, p.trained_dice, width=0.38, label="trained model (leak-free)",
           color="#4c78a8")
    ax.bar(x + 0.19, p.prior_only_dice, width=0.38, label="position prior only",
           color="#f58518")
    ax.set_xticks(x, p.label, rotation=30, ha="right")
    ax.set_ylabel("Dice")
    ax.set_title("(a) a predictor that never sees the image")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.scatter(p.mean_area_frac, p.prior_only_dice, s=34, color="#f58518", zorder=3)
    for _, r in p.iterrows():
        ax.annotate(r.label, (r.mean_area_frac, r.prior_only_dice), xytext=(4, 3),
                    textcoords="offset points", fontsize=7)
    ax.set_xlabel("mean lesion area (fraction of image)")
    ax.set_ylabel("prior-only Dice")
    ax.set_title("(b) the prior works best where\nlesions are large and centred")
    ax.margins(0.18)
    save(fig, "fig4_position_prior")


def fig5_recovery():
    cp = ROOT / "outputs/contamination/changepoint_clinicdb.csv"
    cl = ROOT / "outputs/contamination/sequence_recovery_clinicdb.csv"
    if not cp.exists():
        print("  (skipping fig5: change-point results not found)")
        return
    d = pd.read_csv(cp)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(d.K, d.ARI, "-o", ms=4, color="#4c78a8", label="contiguous change-point")
    if cl.exists():
        c = pd.read_csv(cl)
        best = c.loc[c.groupby("n_clusters").ARI.idxmax()].sort_values("n_clusters")
        ax.plot(best.n_clusters, best.ARI, "-s", ms=3, color="0.6",
                label="unordered clustering")
    ax.axvline(29, color="#d62728", ls="--", lw=1, label="true K = 29")
    ax.set_xlabel("number of recovered sequences")
    ax.set_ylabel("adjusted Rand index")
    ax.set_title("recovering source-video structure\n(CVC-ClinicDB, 29 official sequences)")
    ax.set_xlim(0, 65)
    ax.legend(fontsize=7, loc="lower center", frameon=True, framealpha=0.95,
              edgecolor="none")
    save(fig, "fig5_sequence_recovery")


if __name__ == "__main__":
    print("generating figures")
    fig1_contamination()
    fig2_cvc300()
    fig3_inflation()
    fig4_prior()
    fig5_recovery()
    print(f"\nfigures in {OUT}")
