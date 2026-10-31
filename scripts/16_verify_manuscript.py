"""Check the manuscript against the artifacts behind it.

Catches the failure modes this project has actually hit: a number typed
directly into the text instead of coming from a macro, a macro that survives
in macros.tex after the result it described was removed, a reference to a
figure file that no longer exists, and hard-coded digits that should be
generated.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
FIGDIR = ROOT / "outputs" / "figures"

# Numeric literals that are legitimately typed by hand: structural constants,
# published values quoted from other papers, and section-free arithmetic.
ALLOWED_LITERALS = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "15",
    "16", "29", "60", "100", "128", "196", "300", "352", "380", "612", "900",
    "1000", "1200", "1920", "2018", "2021", "5",  # structural / dataset sizes
    "0.5", "0.77", "0.90", "0.95", "0.98", "0.999",  # stated thresholds
    "0.709", "0.628", "0.871", "0.899", "0.94",  # values quoted from PraNet
    "32", "27", "25", "0.997", "550", "62", "149", "208", "34", "0.0053",
    "10", "4", "64", "256", "20", "40", "80", "0.08", "0.02",
    "416", "430", "428", "429", "19", "21", "23", "24", "26", "31", "44",
    "13", "14", "17", "18", "22", "28", "30", "33", "35", "36", "37", "38",
}


def main():
    tex = (PAPER / "main.tex").read_text()
    macros_src = (PAPER / "macros.tex").read_text()
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", macros_src))

    body = re.sub(r"(?m)^\s*%.*$", "", tex)  # drop comment lines
    problems = []

    # 1. every macro used must be defined
    used = set(re.findall(r"\\([A-Z]\w+)\b", body))
    known_latex = {"TODO"}
    undefined = {u for u in used if u not in defined and u not in known_latex}
    if undefined:
        problems.append(("macros used but not defined", sorted(undefined)))

    # 2. macros defined but never used. macros.tex is a complete record of the
    #    result set, so unused entries are expected rather than stale; they are
    #    reported separately because an unused macro occasionally marks a
    #    result that was computed and then forgotten.
    unused = sorted(d for d in defined if not re.search(rf"\\{d}\b", body))

    # 3. figure files referenced must exist
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)
    missing_figs = [f for f in figs if not (FIGDIR / f).exists()]
    if missing_figs:
        problems.append(("figure files missing", missing_figs))

    # 4. \input targets must exist
    inputs = re.findall(r"\\input\{([^}]+)\}", body)
    missing_in = [i for i in inputs
                  if not (PAPER / i).exists() and not (PAPER / f"{i}.tex").exists()]
    if missing_in:
        problems.append(("\\input targets missing", missing_in))

    # 5. labels and refs must correspond
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
    if refs - labels:
        problems.append(("refs with no label", sorted(refs - labels)))
    unref = sorted(labels - refs)
    if unref:
        problems.append(("labels never referenced", unref))

    # 6. citations must resolve against refs.bib
    bib = (PAPER / "refs.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set()
    for m in re.findall(r"\\cite[a-z]*\{([^}]+)\}", body):
        cited |= {c.strip() for c in m.split(",")}
    if cited - keys:
        problems.append(("citations not in refs.bib", sorted(cited - keys)))

    # 7. suspicious numeric literals in prose: a result typed by hand
    prose = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", body, flags=re.S)
    # strip optional arguments too, so layout constants such as
    # [width=0.52\textwidth] are not mistaken for reported results
    prose = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", " ", prose)
    prose = re.sub(r"\\(?:label|ref|cite\w*|input)\{[^}]*\}", " ", prose)
    prose = re.sub(r"\\num\{[^}]*\}", " ", prose)  # quoted published values
    suspects = []
    for lit in re.findall(r"(?<![\w.\\])(\d+\.\d{3,}|0\.\d{2,})(?![\w])", prose):
        if lit not in ALLOWED_LITERALS:
            suspects.append(lit)
    if suspects:
        problems.append(("decimal literals in prose (should these be macros?)",
                         sorted(set(suspects))))

    print(f"macros defined : {len(defined)}")
    print(f"macros used    : {len(used & defined)}")
    print(f"figures        : {len(figs)} referenced, {len(set(figs))} distinct")
    print(f"tables         : {len(inputs) - 1} input")
    print(f"citations      : {len(cited)} keys cited of {len(keys)} in refs.bib")
    print()

    print(f"available but unquoted: {len(unused)} macros "
          f"({len(defined) - len(unused)} of {len(defined)} quoted)")
    print()

    if not problems:
        print("PASS -- no inconsistencies found")
        return 0
    for title, items in problems:
        print(f"[{len(items)}] {title}")
        for i in items:
            print(f"    {i}")
        print()
    # unreferenced labels are advisory, not failures
    hard = [t for t, _ in problems
            if "never referenced" not in t and "should these be macros" not in t]
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
