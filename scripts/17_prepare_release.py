"""Package the release and check that nothing improper is about to be published.

Produces a single compressed archive of the per-image metrics, then audits
everything git would include for three classes of mistake: credentials,
redistributable image data, and large intermediate caches that should be
regenerated rather than shipped.
"""

import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "outputs" / "release"

# Files whose contents must never be published.
SECRET_NAMES = {"credentials.json", ".env"}
SECRET_PATTERNS = [
    (re.compile(r"rpa_[A-Za-z0-9]{20,}"), "RunPod API key"),
    (re.compile(r"(?i)ssh-(rsa|ed25519)\s+AAAA"), "SSH public key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"(?i)\b(aws_secret_access_key|api[_-]?secret)\b\s*[:=]"), "credential assignment"),
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MAX_FILE_MB = 50


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.splitlines()


def build_metrics_archive():
    RELEASE.mkdir(parents=True, exist_ok=True)
    out = RELEASE / "perimage_metrics.zip"
    csvs = sorted((ROOT / "outputs" / "paper2").glob("perimage_*.csv"))
    if not csvs:
        print("  no per-image CSVs found; skipping archive")
        return None
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for c in csvs:
            z.write(c, f"perimage_metrics/{c.name}")
        readme = (
            "Per-image segmentation metrics for 100 training runs.\n\n"
            "Four arms x five architectures x five folds:\n"
            "  clinicdb_random         random frame splits of CVC-ClinicDB\n"
            "  clinicdb_seqdisjoint    splits respecting the 29 official sequences\n"
            "  kvasir_imagelevel       image-level folds of Kvasir-SEG\n"
            "  kvasir_grouped_tau0.90  folds grouping near-duplicates\n\n"
            "Each row is one image under one trained model. The 'dataset' column\n"
            "gives the test set; '-ext' marks a zero-shot external set. Images are\n"
            "identified by filename so rows can be joined to the original data,\n"
            "which is not redistributed here.\n"
        )
        z.writestr("perimage_metrics/README.txt", readme)
    mb = out.stat().st_size / 1e6
    print(f"  wrote {out.relative_to(ROOT)} ({mb:.1f} MB, {len(csvs)} files)")
    return out


def main():
    print("building release archive")
    build_metrics_archive()

    print("\nauditing what git would publish")
    tracked = git("ls-files")
    untracked = git("ls-files", "--others", "--exclude-standard")
    candidates = [ROOT / f for f in tracked + untracked]
    print(f"  {len(tracked)} tracked + {len(untracked)} untracked-not-ignored "
          f"= {len(candidates)} files")

    problems = []

    # 1. secrets by filename
    named = [c for c in candidates if c.name in SECRET_NAMES]
    if named:
        problems.append(("credential files included",
                         [str(c.relative_to(ROOT)) for c in named]))

    # 2. secrets by content
    leaks = []
    for c in candidates:
        if not c.is_file() or c.stat().st_size > 2_000_000:
            continue
        if c.suffix.lower() in IMAGE_SUFFIXES or c.suffix in {".pdf", ".zip", ".npy"}:
            continue
        try:
            text = c.read_text(errors="ignore")
        except Exception:
            continue
        for pat, what in SECRET_PATTERNS:
            if pat.search(text):
                leaks.append(f"{c.relative_to(ROOT)}: {what}")
    if leaks:
        problems.append(("secret-like content", leaks))

    # 3. non-redistributable image data
    imgs = [c for c in candidates if c.suffix.lower() in IMAGE_SUFFIXES
            and "outputs/figures" not in str(c)]
    if imgs:
        problems.append(("image data (not redistributable)",
                         [str(c.relative_to(ROOT)) for c in imgs[:10]]))

    # 4. oversized files
    big = [(c, c.stat().st_size / 1e6) for c in candidates
           if c.is_file() and c.stat().st_size / 1e6 > MAX_FILE_MB]
    if big:
        problems.append((f"files over {MAX_FILE_MB} MB",
                         [f"{c.relative_to(ROOT)} ({m:.0f} MB)" for c, m in big]))

    total = sum(c.stat().st_size for c in candidates if c.is_file()) / 1e6
    print(f"  total size: {total:.1f} MB")

    if not problems:
        print("\nPASS -- nothing improper staged for release")
        return 0
    print()
    for title, items in problems:
        print(f"[{len(items)}] {title}")
        for i in items:
            print(f"    {i}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
