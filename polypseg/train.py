"""Entry point: paper 1's trainer, with the record-builder widened.

Usage mirrors paper 1 exactly:
    python -m polypseg.train --model unet --splits <csv> --data-root <dir> ...
"""

from ._adapter import apply

if __name__ == "__main__":
    train_mod, _ = apply()
    train_mod.main()
