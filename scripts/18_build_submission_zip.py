"""Build a self-contained LaTeX archive for submission.

The journal compiles the uploaded source itself, so the archive must build
with no reference to anything outside it. Two things in the working copy would
break that:

  * \\graphicspath points at ../outputs/figures/, which will not exist;
  * the bibliography depends on bibtex being run, which the compile service
    may or may not do.

Both are handled here: figures are copied in and the path rewritten to a local
directory, and the pre-built .bbl is shipped alongside refs.bib so the
bibliography resolves whether or not bibtex runs. The working copy in paper/
is left untouched.

The archive is then test-compiled in a clean temporary directory, which is the
only way to be confident it builds the way the journal will build it.
"""

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
FIGS = ROOT / "outputs" / "figures"
OUT = ROOT / "outputs" / "submission"

FIGURES = ["fig1_contamination", "fig2_cvc300_double_counting",
           "fig3_leakage_inflation", "fig4_position_prior",
           "fig5_sequence_recovery"]


def stage(dest: Path):
    """Assemble a self-contained source tree at dest."""
    (dest / "figures").mkdir(parents=True, exist_ok=True)
    (dest / "tables").mkdir(parents=True, exist_ok=True)

    tex = (PAPER / "main.tex").read_text()

    # figures live beside main.tex in the archive
    tex, n = re.subn(r"\\graphicspath\{\{[^}]*\}\}",
                     r"\\graphicspath{{figures/}}", tex)
    if n != 1:
        sys.exit(f"expected exactly one graphicspath, found {n}")

    # xurl is not present in every TeX installation; degrade gracefully rather
    # than fail the journal's compile if it is missing
    tex = tex.replace(
        "\\usepackage{xurl}",
        "\\IfFileExists{xurl.sty}{\\usepackage{xurl}}{\\usepackage{breakurl}}")

    (dest / "main.tex").write_text(tex)
    for f in ("macros.tex", "refs.bib"):
        shutil.copy2(PAPER / f, dest / f)
    if (PAPER / "main.bbl").exists():
        shutil.copy2(PAPER / "main.bbl", dest / "main.bbl")
    else:
        print("  WARNING: main.bbl missing; run latexmk in paper/ first")
    for t in (PAPER / "tables").glob("*.tex"):
        shutil.copy2(t, dest / "tables" / t.name)
    for f in FIGURES:
        src = FIGS / f"{f}.pdf"
        if not src.exists():
            sys.exit(f"missing figure {src}")
        shutil.copy2(src, dest / "figures" / f"{f}.pdf")


def test_compile(src: Path) -> bool:
    """Compile in a pristine copy, as the journal's service would."""
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "build"
        shutil.copytree(src, work)
        r = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"],
            cwd=work, capture_output=True, text=True)
        log = (work / "main.log").read_text(errors="ignore") if (work / "main.log").exists() else ""
        pdf = work / "main.pdf"
        errors = [l for l in log.splitlines() if l.startswith("! ")]
        undef = [l for l in log.splitlines()
                 if "Undefined control sequence" in l
                 or "undefined references" in l.lower()
                 or "Citation" in l and "undefined" in l]
        missing_fig = [l for l in log.splitlines() if "File" in l and "not found" in l]
        pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)

        print(f"  latexmk exit      : {r.returncode}")
        print(f"  pdf produced      : {pdf.exists()}")
        print(f"  pages             : {pages.group(1) if pages else 'n/a'}")
        print(f"  '!' errors        : {len(errors)}")
        print(f"  undefined refs    : {len(undef)}")
        print(f"  missing files     : {len(missing_fig)}")
        for l in (errors + undef + missing_fig)[:5]:
            print(f"      {l.strip()[:96]}")
        ok = pdf.exists() and not errors and not missing_fig
        if ok and pages:
            shutil.copy2(pdf, OUT / "compiled_preview.pdf")
        return ok


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    staging = OUT / "latex_source"
    if staging.exists():
        shutil.rmtree(staging)
    print("staging self-contained source")
    stage(staging)
    for p in sorted(staging.rglob("*")):
        if p.is_file():
            print(f"    {p.relative_to(staging)}  ({p.stat().st_size/1024:.0f} KB)")

    print("\ntest-compiling in a clean directory")
    ok = test_compile(staging)

    if not ok:
        print("\nFAIL -- archive would not compile; not packaging")
        return 1

    zip_path = OUT / "npj_submission_latex.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(staging.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(staging))
    print(f"\nPASS -- wrote {zip_path.relative_to(ROOT)} "
          f"({zip_path.stat().st_size/1024:.0f} KB)")
    print(f"       preview PDF at {(OUT/'compiled_preview.pdf').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
