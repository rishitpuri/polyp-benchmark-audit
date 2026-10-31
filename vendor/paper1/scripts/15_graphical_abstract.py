"""Graphical abstract.

Appears in the table of contents at small size, so it carries one message
only: per-lesion detection collapses on lesions under 1% of the frame, and
the benchmark the field trains and tests on contains almost none of them.
Deliberately larger type and fewer elements than the in-paper figures.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "polypseg" / "outputs"
DEST = OUT / "figures_final" / "graphical_abstract"

MODELS = ["unet", "unetpp", "segformer", "segformer_b2", "unet_pvt"]

plt.rcParams.update({
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})


def main():
    fr = []
    for m in MODELS:
        for k in range(5):
            p = OUT / "cv2" / f"perimage_{m}_seed42_fold{k}.csv"
            if p.exists():
                fr.append(pd.read_csv(p))
    cv = pd.concat(fr, ignore_index=True)

    ext = pd.read_csv(OUT / "external_metadata.csv")
    ext["name"] = ext["name"].astype(str)
    kv = pd.read_csv(OUT / "metadata.csv")[["name", "area_frac"]]

    e = cv[cv.dataset != "kvasir-seg"].copy()
    e["name"] = e["name"].astype(str)
    e = e.merge(ext[["dataset", "name", "area_frac"]], on=["dataset", "name"], how="left")
    k = cv[cv.dataset == "kvasir-seg"].merge(kv, on="name", how="left")
    allimg = pd.concat([e, k], ignore_index=True)
    allimg["bin"] = pd.cut(allimg.area_frac, [0, 0.01, 0.05, 1.0],
                           labels=["<1%", "1–5%", ">5%"])

    det = [allimg[allimg.bin == b].det_rate_iou05.mean() for b in ["<1%", "1–5%", ">5%"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    colors = ["#B00000", "#8899aa", "#8899aa"]
    bars = axes[0].bar(["<1%", "1–5%", ">5%"], det, color=colors, width=0.62)
    for b, v in zip(bars, det):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.2f}",
                     ha="center", fontsize=14, fontweight="bold")
    axes[0].axhline(0.5, ls=":", c="0.35", lw=1.4)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Lesions detected\n(IoU $\\geq$ 0.5)", fontsize=12)
    axes[0].set_xlabel("Lesion size (% of image area)", fontsize=12)
    axes[0].set_title("Half of clinically small lesions\nare missed by every architecture",
                      fontsize=13, fontweight="bold")

    names = ["Kvasir-SEG", "CVC-\nClinicDB", "CVC-\n300",
             "CVC-\nColonDB", "ETIS-\nLarib"]
    src = ["kvasir-seg", "cvc-clinicdb", "cvc-300", "cvc-colondb", "etis-larib"]
    pct = []
    for s in src:
        a = kv.area_frac if s == "kvasir-seg" else ext[ext.dataset == s].area_frac
        pct.append(100 * (a < 0.01).mean())
    bars = axes[1].bar(names, pct, color=["#B00000"] + ["#8899aa"] * 4, width=0.62)
    for b, v in zip(bars, pct):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%",
                     ha="center", fontsize=12)
    axes[1].set_ylabel("Images with a lesion\n<1% of the frame", fontsize=12)
    axes[1].set_ylim(0, max(pct) * 1.22)
    axes[1].set_title("…and the benchmark contains\nalmost none of them",
                      fontsize=13, fontweight="bold")
    axes[1].tick_params(axis="x", labelsize=10)
    # Mark which bar is the field's standard benchmark without crowding the
    # tick labels into one another.
    axes[1].annotate("the field's\nbenchmark", xy=(0, pct[0] + 2.6),
                     xytext=(0.75, max(pct) * 0.42), fontsize=10, color="#B00000",
                     ha="center",
                     arrowprops=dict(arrowstyle="->", color="#B00000", lw=1.3,
                                     shrinkA=2, shrinkB=4))

    fig.tight_layout(w_pad=2.2)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    for ext_ in ("png", "pdf", "tif"):
        try:
            fig.savefig(f"{DEST}.{ext_}", dpi=300, bbox_inches="tight")
        except Exception as exc:                      # TIFF needs Pillow
            print(f"  ({ext_} skipped: {exc})")
    plt.close(fig)
    print(f"wrote {DEST}.png / .pdf")
    print(f"detection by size: {[round(d, 3) for d in det]}")
    print(f"small-lesion share by dataset: {[round(p, 1) for p in pct]}")


if __name__ == "__main__":
    main()
