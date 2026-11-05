"""Emit LaTeX macros for every number quoted in the manuscript.

Same discipline as paper 1: the text never hard-codes a result. Each value is
written here from the result CSVs and pulled into main.tex via \\input, so the
manuscript cannot drift from the experiments that produced it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper"
SIGMA_RUN = 0.0053


def sci_math(p, underflowed, log10_p):
    """Format a p-value for math mode, without a leading dollar sign.

    When the exact value underflows, fall back to the order of magnitude
    derived from the normal approximation rather than printing zero.
    """
    if underflowed or p <= 0:
        return f"10^{{{log10_p:.0f}}}"
    exponent = int(np.floor(np.log10(p)))
    mantissa = p / (10 ** exponent)
    return f"{mantissa:.2f} \\times 10^{{{exponent}}}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = {}

    def put(key, value, fmt="{}"):
        m[key] = fmt.format(value)

    # ---- contamination ----
    idx = pd.read_csv(ROOT / "outputs/contamination/image_index.csv")
    red = pd.read_csv(ROOT / "outputs/contamination/within_dataset_redundancy.csv").set_index("dataset")
    summ = pd.read_csv(ROOT / "outputs/contamination/cross_dataset_summary.csv")

    # Kvasir-Sessile is drawn from Kvasir-SEG, so it is excluded from the
    # suite total to avoid double-counting its images.
    SUITE = ["kvasir-seg", "cvc-clinicdb", "cvc-colondb", "cvc-300", "etis-larib"]
    put("NimagesTotal", f"{int(idx.dataset.isin(SUITE).sum()):,}")
    for ds, short in [("cvc-colondb", "Colon"), ("etis-larib", "Etis"),
                      ("kvasir-seg", "Kvasir"), ("cvc-clinicdb", "Clinic"),
                      ("cvc-300", "CVCThree"), ("kvasir-sessile", "Sessile")]:
        g = idx[idx.dataset == ds]
        put(f"N{short}", len(g))
        put(f"NUnique{short}", g.pixel_md5.nunique())
        put(f"NDupes{short}", len(g) - g.pixel_md5.nunique())
        put(f"NearDupNinety{short}", f"{100*red.loc[ds,'frac_nn_r>0.90']:.1f}")
        put(f"NearDupNinetyFive{short}", f"{100*red.loc[ds,'frac_nn_r>0.95']:.1f}")
        put(f"NearDupNinetyEight{short}", f"{100*red.loc[ds,'frac_nn_r>0.98']:.1f}")

    row = summ[(summ["query"] == "cvc-300") & (summ.reference == "cvc-colondb")].iloc[0]
    put("CVCThreeMatched", int(row["n_hamming<=0"]))

    # Kvasir-Sessile: detected by the perceptual signals but invisible to
    # exact pixel hashing, because the images were re-encoded between releases.
    sr = summ[(summ["query"] == "kvasir-sessile") & (summ.reference == "kvasir-seg")].iloc[0]
    put("SessileN", int(sr["n_query"]))
    put("SessileMatchedCorr", int(sr["n_r>0.999"]))
    put("SessileMatchedPhash", int(sr["n_hamming<=5"]))
    put("SessileMatchedPhashExact", int(sr["n_hamming<=0"]))
    sess = idx[idx.dataset == "kvasir-sessile"]
    kv = set(idx[idx.dataset == "kvasir-seg"].pixel_md5)
    put("SessileMatchedPixel", int(sess.pixel_md5.isin(kv).sum()))

    ident = pd.read_csv(ROOT / "outputs/rescore/cvc300_identity_check.csv")
    put("IdentityMaxDiff", f"{(ident.cvc300 - ident.colondb_149_208).abs().max():.1e}")
    put("BlockDice", f"{ident.colondb_149_208.mean():.4f}")
    put("RestDice", f"{ident.colondb_rest.mean():.4f}")
    put("BlockGap", f"{ident.colondb_149_208.mean() - ident.colondb_rest.mean():+.4f}")

    # ---- re-scoring ----
    shift = pd.read_csv(ROOT / "outputs/rescore/external_average_shift.csv")
    put("RescoreMinShift", f"{shift.delta.abs().min():.4f}")
    put("RescoreMaxShift", f"{shift.delta.abs().max():.4f}")
    put("RescoreMinSigma", f"{shift.delta.abs().min()/SIGMA_RUN:.1f}")
    put("RescoreMaxSigma", f"{shift.delta.abs().max()/SIGMA_RUN:.1f}")
    put("RescoreNShifted", int((shift.delta.abs() > SIGMA_RUN).sum()))
    put("RescoreNModels", len(shift))
    top_pub = shift.sort_values("rank_published").iloc[0]
    top_cor = shift.sort_values("rank_corrected").iloc[0]
    put("TopPublished", top_pub.model.replace("_", "\\_"))
    put("TopCorrected", top_cor.model.replace("_", "\\_"))
    pairs = pd.read_csv(ROOT / "outputs/rescore/pairwise_gaps.csv")
    put("NSignFlips", int(pairs.sign_flip.sum()))
    put("NPairs", len(pairs))

    # ---- leakage inflation ----
    for key, tag in (("clinicdb", "Clinic"), ("kvasir", "Kvasir")):
        pooled = pd.read_csv(ROOT / f"outputs/inflation/inflation_pooled_{key}.csv")
        ind = pooled[pooled.dataset.str.contains("in-domain")].iloc[0]
        put(f"Inflation{tag}", f"{ind.inflation:.4f}")
        put(f"Inflation{tag}Lo", f"{ind.ci_lo:.4f}")
        put(f"Inflation{tag}Hi", f"{ind.ci_hi:.4f}")
        put(f"Inflation{tag}Sigma", f"{ind.in_sigma_run:.1f}")
        put(f"Inflation{tag}Random", f"{ind['random']:.4f}")
        put(f"Inflation{tag}Clean", f"{ind.seq_disjoint:.4f}")
        # ind.n_images counts paired (architecture, image) observations, which
        # is five times the number of frames; both are needed and they must
        # not be confused in the text.
        put(f"Inflation{tag}Pairs", f"{int(ind.n_images):,}")
        put(f"Inflation{tag}N", f"{int(ind.n_images) // 5:,}")
        # test statistics; the exact p underflows for the ClinicDB comparison,
        # so log10(p) from the normal approximation is carried alongside
        put(f"Inflation{tag}W", f"{int(ind.wilcoxon_W):,}")
        put(f"Inflation{tag}Z", f"{ind.wilcoxon_z:.1f}")
        put(f"Inflation{tag}LogP", f"{ind.wilcoxon_log10_p:.0f}")
        put(f"Inflation{tag}Nonzero", f"{int(ind.n_nonzero_pairs):,}")
        put(f"Inflation{tag}P", sci_math(ind.wilcoxon_p, ind.p_underflowed,
                                        ind.wilcoxon_log10_p))
        ext = pooled[~pooled.dataset.str.contains("in-domain")]
        put(f"Ext{tag}Min", f"{ext.inflation.min():.4f}")
        put(f"Ext{tag}Max", f"{ext.inflation.max():.4f}")
        bym = pd.read_csv(ROOT / f"outputs/inflation/inflation_by_model_{key}.csv")
        bi = bym[bym.dataset.str.contains("in-domain")]
        put(f"Inflation{tag}ModelMin", f"{bi.inflation.min():.4f}")
        put(f"Inflation{tag}ModelMax", f"{bi.inflation.max():.4f}")
        put(f"Inflation{tag}SigmaMin", f"{bi.in_sigma_run.min():.1f}")
        put(f"Inflation{tag}SigmaMax", f"{bi.in_sigma_run.max():.1f}")
        rk = pd.read_csv(ROOT / f"outputs/inflation/ranking_shift_{key}.csv")
        put(f"RankMoved{tag}", int((rk.moved != 0).sum()))

    # leakage-proper after subtracting the diversity estimate
    ic = float(m["InflationClinic"])
    div = pd.read_csv(ROOT / "outputs/inflation/inflation_pooled_clinicdb.csv")
    div = div[~div.dataset.str.contains("in-domain")].inflation.mean()
    put("DiversityEstimate", f"{div:.4f}")
    put("InflationAdjusted", f"{ic - div:.4f}")
    put("InflationAdjustedSigma", f"{(ic - div)/SIGMA_RUN:.1f}")

    # ---- sequence recovery ----
    cp = pd.read_csv(ROOT / "outputs/contamination/changepoint_clinicdb.csv")
    at29 = cp[cp.K == 29].iloc[0]
    put("RecoveryARI", f"{at29.ARI:.3f}")
    put("RecoveryBoundaryF", f"{at29.bF1_tol2:.3f}")
    cl = pd.read_csv(ROOT / "outputs/contamination/sequence_recovery_clinicdb.csv")
    put("ClusterARI", f"{cl.ARI.max():.3f}")
    kr = pd.read_csv(ROOT / "outputs/contamination/kratio_calibration_clinicdb.csv")
    safe = kr[kr.leak_safe]
    put("SafeKRatio", f"{safe.K_ratio.max():.3f}")
    put("SafeHomogeneity", f"{safe.sort_values('K_ratio').iloc[-1].homogeneity:.3f}")

    # ---- position prior ----
    pr = pd.read_csv(ROOT / "outputs/prior/prior_summary.csv").set_index("dataset")
    for ds, short in [("kvasir-seg", "Kvasir"), ("cvc-clinicdb", "Clinic"),
                      ("cvc-colondb", "Colon"), ("etis-larib", "Etis")]:
        put(f"Prior{short}", f"{pr.loc[ds,'prior_only_dice']:.4f}")
        put(f"PriorTrained{short}", f"{pr.loc[ds,'trained_dice']:.4f}")
        put(f"PriorPct{short}", f"{pr.loc[ds,'pct_of_trained']:.0f}")

    # ---- released PraNet checkpoint ----
    pn_path = ROOT / "outputs/pranet/pranet_summary.csv"
    if pn_path.exists():
        pn = pd.read_csv(pn_path).iloc[0]
        put("PraNetCVCThree", f"{pn.cvc300:.4f}")
        put("PraNetBlock", f"{pn.colondb_block:.4f}")
        put("PraNetBlockDiff", f"{abs(pn.cvc300 - pn.colondb_block):.1e}")
        put("PraNetRest", f"{pn.colondb_rest:.4f}")
        put("PraNetBlockGap", f"{pn.colondb_block - pn.colondb_rest:+.4f}")
        put("PraNetColon", f"{pn.colondb_all:.4f}")
        put("PraNetEtis", f"{pn.etis:.4f}")
        put("PraNetAvgThree", f"{pn.avg_unseen_three:.4f}")
        put("PraNetAvgTwo", f"{pn.avg_unseen_two:.4f}")
        # pn["shift"] rather than pn.shift: the latter resolves to the pandas
        # Series.shift method, not to the column of that name
        put("PraNetShift", f"{pn['shift']:+.4f}")
        put("PraNetShiftSigma", f"{abs(pn['shift'])/SIGMA_RUN:.1f}")
        put("PraNetClinicAll", f"{pn.clinicdb_all_not_a_test_score:.4f}")

    put("SigmaRun", f"{SIGMA_RUN:.4f}")
    put("NRuns", 100)

    lines = ["% Generated by scripts/13_make_macros.py -- do not edit by hand.",
             "% Every number quoted in the manuscript is defined here from the result CSVs.", ""]
    for k, v in sorted(m.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (OUT / "macros.tex").write_text("\n".join(lines) + "\n")

    print(f"wrote {len(m)} macros to {OUT/'macros.tex'}")
    for k in ("InflationClinic", "InflationClinicSigma", "InflationAdjusted",
              "InflationAdjustedSigma", "InflationKvasir", "InflationKvasirSigma",
              "CVCThreeMatched", "IdentityMaxDiff", "NDupesColon", "RecoveryARI",
              "PriorKvasir", "TopPublished", "TopCorrected"):
        print(f"  \\{k} = {m[k]}")


if __name__ == "__main__":
    main()
