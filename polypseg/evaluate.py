"""Entry point: paper 1's evaluator, with the record-builder widened.

Paper 1 only ever trained in-domain on Kvasir-SEG, so its evaluator labels the
in-domain rows with the literal string "kvasir-seg" (evaluate.py line 78).
Paper 2 also trains on CVC-ClinicDB, where that label would be wrong, and
would additionally collide with Kvasir-SEG when the latter is used as an
external set.

Rather than edit the vendored evaluator, the in-domain rows are relabelled
afterwards when POLYPSEG_INDOMAIN_NAME is set. External sets are passed under
distinct names so no collision is possible.
"""

import os
import sys
from pathlib import Path

import pandas as pd

from ._adapter import apply


def _arg(argv, flag):
    return Path(argv[argv.index(flag) + 1]) if flag in argv else None


if __name__ == "__main__":
    _, evaluate_mod = apply()
    evaluate_mod.main()

    name = os.environ.get("POLYPSEG_INDOMAIN_NAME")
    out = _arg(sys.argv, "--out")
    if name and out and out.exists():
        df = pd.read_csv(out)
        n = int((df["dataset"] == "kvasir-seg").sum())
        df.loc[df["dataset"] == "kvasir-seg", "dataset"] = name
        df.to_csv(out, index=False)
        print(f"[adapter] relabelled {n} in-domain rows 'kvasir-seg' -> '{name}'")
