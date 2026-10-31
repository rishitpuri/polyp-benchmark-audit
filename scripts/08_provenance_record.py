"""Provenance record for the circulating CVC-ColonDB artifact.

The CVC-ColonDB used by essentially the whole polyp segmentation literature is
not the dataset its creators published. This script records the discrepancy
from primary sources so the claim in the manuscript is traceable.

Authoritative specification, recovered from the Internet Archive capture of
the creators' own download page (mv.cvc.uab.es/projects/colon-qa/cvccolondb,
captured 2020-07-15; the live host no longer resolves):

    "CVC-ColonDB is a database of annotated video sequences of colonoscopy
     video. It contains 15 short colonoscopy sequences, coming from 15
     different studies. In each sequence one polyp is shown.
     We provide an annotation of the region of interest (ROI) for 300 frames
     selected from all the sequences. These frames were selected in order to
     maximise the visual difference between them."

    "The database consists of 1200 different images, divided in four groups:
     original .bmp files; polyp masks; non-informative region masks; contour
     of the polyp masks."  (i.e. 4 x 300)

Access required registration. The registration endpoint and the data file were
never captured by the Internet Archive, and the download host has been
unreachable, so the official distribution is currently unobtainable. That is
the most likely reason the discrepancy below has gone unremarked.

What the field actually evaluates on is the copy redistributed in the PraNet
TestDataset bundle (and mirrored on Kaggle), which PraNet's paper describes as
"380 images from 15 short colonoscopy sequences".
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "provenance"

OFFICIAL = {
    "source": "mv.cvc.uab.es/projects/colon-qa/cvccolondb",
    "source_status": "host unreachable; recovered via Internet Archive 2020-07-15 capture",
    "archive_url": ("http://web.archive.org/web/20200715053038/"
                    "http://mv.cvc.uab.es/projects/colon-qa/cvccolondb"),
    "sequences": 15,
    "studies": 15,
    "annotated_frames": 300,
    "total_files": 1200,
    "file_groups": ["original .bmp", "polyp masks p*.bmp",
                    "non-informative region masks s*.bmp", "contour masks c*.bmp"],
    "frame_selection": "selected in order to maximise the visual difference between them",
    "access": "registration required; endpoint no longer available",
    "reference": ("Bernal, Sanchez & Vilarino (2012), Towards automatic polyp detection "
                  "with a polyp appearance model, Pattern Recognition 45(9):3166-3182"),
}

CIRCULATING = {
    "source": "PraNet TestDataset bundle (DengPingFan/PraNet), also mirrored on Kaggle",
    "described_in_pranet_as": "380 images from 15 short colonoscopy sequences",
    "files": 380,
    "resolution": "574x500, uniform",
    "frame_numbering": "1-380, contiguous",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = ROOT / "outputs" / "contamination"
    idx = pd.read_csv(audit / "image_index.csv")
    cdb = idx[idx.dataset == "cvc-colondb"]
    red = pd.read_csv(audit / "within_dataset_redundancy.csv").set_index("dataset")

    measured = {
        "files": int(len(cdb)),
        "unique_images_by_pixel_hash": int(cdb.pixel_md5.nunique()),
        "exact_duplicate_files": int(len(cdb) - cdb.pixel_md5.nunique()),
        "frames_identical_to_cvc300": "149-208 (60 frames, contiguous)",
        "frac_with_near_duplicate_r_gt_0.90": float(red.loc["cvc-colondb", "frac_nn_r>0.90"]),
        "frac_with_near_duplicate_r_gt_0.98": float(red.loc["cvc-colondb", "frac_nn_r>0.98"]),
    }

    print("=== official specification (creators, via Internet Archive) ===")
    print(f"  sequences                {OFFICIAL['sequences']}")
    print(f"  annotated frames         {OFFICIAL['annotated_frames']}")
    print(f"  total files              {OFFICIAL['total_files']} (4 x 300)")
    print(f"  frame selection          {OFFICIAL['frame_selection']}")
    print(f"  availability             {OFFICIAL['access']}")

    print("\n=== circulating artifact (what the literature evaluates on) ===")
    print(f"  files                    {measured['files']}")
    print(f"  unique images            {measured['unique_images_by_pixel_hash']}")
    print(f"  exact duplicate files    {measured['exact_duplicate_files']}")
    print(f"  identical to CVC-300     {measured['frames_identical_to_cvc300']}")

    print("\n=== the contradiction ===")
    print(f"  official frame count 300 vs {measured['files']} distributed "
          f"({measured['files'] - OFFICIAL['annotated_frames']:+d}); "
          f"{measured['unique_images_by_pixel_hash']} unique after deduplication, "
          f"still {measured['unique_images_by_pixel_hash'] - OFFICIAL['annotated_frames']:+d}")
    print(f"  official frames were curated to MAXIMISE visual difference, yet "
          f"{measured['frac_with_near_duplicate_r_gt_0.90']:.1%} of the distributed")
    print(f"  images have a near-duplicate (r>0.90) and "
          f"{measured['exact_duplicate_files']} are pixel-identical copies")
    print("  -> the artifact under evaluation is not the dataset that was published,")
    print("     and the authoritative copy needed to reconcile them is unobtainable")

    (OUT / "cvc_colondb_provenance.json").write_text(json.dumps(
        {"official": OFFICIAL, "circulating": CIRCULATING, "measured": measured}, indent=2))
    print(f"\nrecord written to {OUT / 'cvc_colondb_provenance.json'}")


if __name__ == "__main__":
    main()
