"""Evaluate the released PraNet checkpoint on the suite it defined.

Everything else in this study measures our own five architectures. That leaves
one gap a reader will reasonably press on: whether the defects we document
affect the model the field actually cites, rather than only our proxies for
it. PraNet published both the evaluation protocol in question and pretrained
weights, so its own checkpoint can be run over the same images without any
retraining and without our recipe entering the comparison.

The checkpoint is used exactly as released. The point is not to reproduce
PraNet's reported numbers to the decimal, which would depend on preprocessing
details we cannot fully recover, but to establish on the canonical model that
(i) CVC-300 carries no information beyond CVC-ColonDB, and (ii) the block it
duplicates is easier than the rest of its parent, so counting it separately
biases an average upward.

Requires the PraNet repository and checkpoint; see PRANET_DIR below.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "pranet"
PRANET_DIR = Path("/tmp/pranet_probe")
CKPT = PRANET_DIR / "snapshots/PraNet_Res2Net/PraNet-19.pth"

SIZE = 352
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DATASETS = {
    "cvc-colondb": DATA / "external/cvc-colondb",
    "cvc-300": DATA / "external/cvc-300",
    "etis-larib": DATA / "external/etis-larib",
    "cvc-clinicdb": DATA / "external/cvc-clinicdb",
}


def build_model(device):
    sys.path.insert(0, str(PRANET_DIR))
    import lib.PraNet_Res2Net as pranet_mod

    # The backbone constructor loads ImageNet weights from an absolute path on
    # the original authors' filesystem. Those weights are irrelevant here: the
    # released PraNet checkpoint replaces every parameter immediately below,
    # so the backbone is instantiated without them.
    original = pranet_mod.res2net50_v1b_26w_4s
    pranet_mod.res2net50_v1b_26w_4s = lambda pretrained=False, **kw: original(
        pretrained=False, **kw)
    model = pranet_mod.PraNet()
    pranet_mod.res2net50_v1b_26w_4s = original

    # released checkpoints predate weights_only defaulting to True
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            raise SystemExit(f"checkpoint does not match the model: {missing[:5]}")
    return model.to(device).eval()


def dice(pred, gt):
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2.0 * (pred * gt).sum() / s)


@torch.no_grad()
def evaluate(model, root, device):
    img_dir, mask_dir = root / "images", root / "masks"
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in
             {".png", ".jpg", ".jpeg", ".tif", ".bmp"}}
    rows = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}:
            continue
        if p.stem not in masks:
            continue
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            x = np.asarray(rgb.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)

        # PraNet returns coarse-to-fine side outputs; the last is the prediction
        out = model(t)
        logits = out[-1] if isinstance(out, (tuple, list)) else out
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        # PraNet's own evaluation min-max normalises the map before thresholding
        prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
        pred = (prob > 0.5).astype(np.float32)

        with Image.open(masks[p.stem]) as mk:
            gt = (np.asarray(mk.convert("L"), dtype=np.uint8) > 127).astype(np.float32)
        if gt.shape != pred.shape:
            gt = np.asarray(Image.fromarray((gt * 255).astype(np.uint8)).resize(
                (pred.shape[1], pred.shape[0]), Image.NEAREST), dtype=np.uint8) > 127
            gt = gt.astype(np.float32)
        rows.append({"name": p.stem, "dice": dice(pred, gt)})
    return pd.DataFrame(rows)


def main():
    if not CKPT.exists():
        raise SystemExit(f"checkpoint not found at {CKPT}")
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    model = build_model(device)

    frames = {}
    for name, root in DATASETS.items():
        df = evaluate(model, root, device)
        df["dataset"] = name
        df.to_csv(OUT / f"pranet_perimage_{name}.csv", index=False)
        frames[name] = df
        print(f"  {name:<14} n={len(df):>4}  mean Dice {df.dice.mean():.4f}")

    cd = frames["cvc-colondb"].copy()
    cd["frame"] = [int(re.sub(r"\D", "", s)) for s in cd.name]
    blk = cd[(cd.frame >= 149) & (cd.frame <= 208)]
    rest = cd[(cd.frame < 149) | (cd.frame > 208)]
    c3 = frames["cvc-300"]

    print("\n=== is CVC-300 informative beyond CVC-ColonDB? ===")
    print(f"  CVC-300 alone                      {c3.dice.mean():.4f}  (n={len(c3)})")
    print(f"  CVC-ColonDB frames 149-208         {blk.dice.mean():.4f}  (n={len(blk)})")
    print(f"  |difference|                       {abs(c3.dice.mean()-blk.dice.mean()):.6f}")
    print(f"  CVC-ColonDB excluding that block   {rest.dice.mean():.4f}  (n={len(rest)})")
    print(f"  the duplicated block is            {blk.dice.mean()-rest.dice.mean():+.4f} easier")

    # PraNet trains on 550 of the 612 CVC-ClinicDB images, so that dataset is
    # not an unseen test set for this checkpoint and must not enter the
    # generalisation average. Including it would repeat the very error under
    # discussion; it is reported separately below as an illustration of one.
    UNSEEN = ("cvc-colondb", "cvc-300", "etis-larib")
    pub = np.mean([frames[d].dice.mean() for d in UNSEEN])
    cor = np.mean([frames[d].dice.mean() for d in UNSEEN if d != "cvc-300"])
    print("\n=== PraNet's three unseen datasets, with and without the duplicate ===")
    print(f"  mean over the three as reported    {pub:.4f}")
    print(f"  mean over the two that are distinct {cor:.4f}")
    print(f"  shift                              {cor-pub:+.4f}")
    print("  -- the protocol's 'three unseen datasets' are two")

    print("\n=== aside: evaluating on all of CVC-ClinicDB ===")
    print(f"  all {len(frames['cvc-clinicdb'])} images        "
          f"{frames['cvc-clinicdb'].dice.mean():.4f}")
    print("  PraNet trained on 550 of these, so this figure is not a test score;")
    print("  it is quoted only to show how large the gap to a held-out score can be.")

    pd.DataFrame([{
        "cvc300": c3.dice.mean(), "colondb_block": blk.dice.mean(),
        "colondb_rest": rest.dice.mean(), "colondb_all": cd.dice.mean(),
        "etis": frames["etis-larib"].dice.mean(),
        "clinicdb_all_not_a_test_score": frames["cvc-clinicdb"].dice.mean(),
        "avg_unseen_three": pub, "avg_unseen_two": cor, "shift": cor - pub,
    }]).to_csv(OUT / "pranet_summary.csv", index=False)
    print(f"\nartifacts in {OUT}")


if __name__ == "__main__":
    main()
