"""Adapter over paper 1's training code.

Paper 1's `kvasir_records` hardcodes the Kvasir-SEG layout, where image and
mask share a stem and both are JPEG:

    root/images/{name}.jpg , root/masks/{name}.jpg

Paper 2 also trains on CVC-ClinicDB, whose files are PNG, so that function
cannot be reused unchanged. Rather than edit paper 1's code -- which would
break the claim that both papers train under one identical recipe, the very
thing that makes the inflation comparison interpretable -- we widen the record
builder here and leave `vendor/paper1/` byte-identical to the released
version.

The widening is strictly additive. If a splits CSV carries explicit
`image_file` and `mask_file` columns they are used verbatim; otherwise the
call falls through to paper 1's original function, so every paper 1 split
file reproduces paper 1's behaviour exactly.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "paper1"

def _fallback_records(root: Path, splits_csv: Path, split: str):
    """Paper 1's original behaviour, reproduced for the no-torch code path."""
    df = pd.read_csv(splits_csv)
    df = df[df["split"] == split]
    return [
        (r["name"], Path(root) / "images" / f"{r['name']}.jpg",
         Path(root) / "masks" / f"{r['name']}.jpg")
        for _, r in df.iterrows()
    ]


_original = None  # set by apply() to paper 1's own implementation


def records(root: Path, splits_csv: Path, split: str):
    """Paper 1's record builder, extended to explicit filenames.

    Importable without torch so the join logic can be tested on any machine;
    the vendored modules are only pulled in by `apply()`. When running for
    real, splits lacking the explicit columns are delegated to paper 1's own
    function, so its behaviour is used rather than re-implemented.
    """
    df = pd.read_csv(splits_csv)
    df = df[df["split"] == split]
    if {"image_file", "mask_file"}.issubset(df.columns):
        return [
            (str(r["name"]),
             Path(root) / "images" / str(r["image_file"]),
             Path(root) / "masks" / str(r["mask_file"]))
            for _, r in df.iterrows()
        ]
    return (_original or _fallback_records)(root, splits_csv, split)


def apply():
    """Rebind the record builder in every module that imported it by name."""
    global _original
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from src import data as p1_data
    from src import evaluate as p1_evaluate
    from src import train as p1_train

    if _original is None:
        _original = p1_data.kvasir_records
    p1_data.kvasir_records = records
    p1_train.kvasir_records = records
    p1_evaluate.kvasir_records = records
    return p1_train, p1_evaluate
