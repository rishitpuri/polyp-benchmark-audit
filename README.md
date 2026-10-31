# Duplicate images and leaking splits in public polyp segmentation benchmarks

Code, audit toolkit, leakage-free splits and per-image results for an
image-level audit of the public polyp segmentation evaluation suite.

**Headline results**

1. All 60 **CVC-300** images are pixel-identical to **CVC-ColonDB** frames
   149–208, so the two are not independent test sets. Confirmed three ways:
   MD5 of decoded pixels, perceptual hash at zero distance, and mean Dice
   agreeing to 1.9e-06 across five independently trained architectures.
   The duplicated block is **+0.13 Dice easier** than the rest of its parent,
   so counting it separately biases an average upward.
2. **34 of 380** CVC-ColonDB files and 3 of 196 ETIS-Larib files are exact
   duplicates of other files in the same folder.
3. **Kvasir-Sessile is contained in Kvasir-SEG** (196/196 at correlation
   > 0.999) but **0/196 by exact pixel hash**, because the images were
   re-encoded between releases. A checksum-based audit reports it clean.
4. The distributed **CVC-ColonDB cannot be reconciled with its specification**
   of 300 frames "selected in order to maximise the visual difference between
   them": it ships 380 files, 346 unique, with 85.8% carrying a near-duplicate.
   The authoritative copy is unobtainable, so the discrepancy cannot be
   resolved from primary sources.
5. On the **released PraNet checkpoint**, the protocol's three unseen datasets
   are two: the mean over them falls from 0.7429 to 0.6760, a shift of
   **−0.0669 Dice = 12.6 σ_run**.
6. Random frame splits of CVC-ClinicDB **leak in every case**, inflating
   in-domain Dice by **+0.0970 (18.3 σ_run)**, or +0.0701 (13.2 σ_run) after
   adjusting for training diversity. Kvasir-SEG inflates by only
   **+0.0059 (1.1 σ_run)**.
7. A predictor that **never sees the image** reaches **0.3963** Dice on
   Kvasir-SEG and 0.2982 on CVC-ClinicDB.
8. Source-video structure is recoverable from pixels at **ARI 0.997** against
   CVC-ClinicDB's 29 official sequences.

Effects are reported in units of σ_run = 0.0053 Dice, the run-to-run noise
floor measured for this training recipe in the companion study.

## Layout

```
scripts/
  01_contamination_audit.py   image index, duplicate and near-duplicate matrices
  02_sequence_recovery.py     unordered clustering baseline (superseded by 03)
  03_sequence_changepoint.py  contiguous change-point recovery, validated
  04_k_selection.py           K selection: slope and calibrated rules
  05_k_selection_adaptive.py  fixed-point variant (negative result, retained)
  06_leak_safe_groups.py      leak-safe grouping and canonical ClinicDB folds
  07_rescore_benchmark.py     re-scoring with the duplication removed
  08_provenance_record.py     CVC-ColonDB specification vs distributed artifact
  09_make_splits.py           training splits for the four experimental arms
  10_analyze_inflation.py     paired random-vs-leak-free comparison
  11_position_prior.py        content-free position-prior baseline
  12_figures.py               all manuscript figures
  13_make_macros.py           LaTeX macros for every quoted number
  14_pranet_eval.py           released PraNet checkpoint on the same images
  15_make_tables.py           manuscript tables
  16_verify_manuscript.py     checks the text against the artifacts
  pod_setup.sh / pod_sync.sh  GPU bootstrap and result retrieval
polypseg/                     adapter over the companion study's training code
vendor/paper1/                companion study's code, unmodified
outputs/                      all results (see below)
paper/                        manuscript, macros and tables
run_paper2.sh                 the 100 training runs, resumable
```

`vendor/paper1/` is byte-identical to the released companion code. The adapter
in `polypseg/` widens its record builder to handle non-JPEG datasets and
delegates to the original function otherwise, so both studies share one
training recipe. Nothing in `vendor/` is modified.

## Released results

| path | contents |
|---|---|
| `outputs/contamination/image_index.csv` | every image with pixel MD5 and perceptual hash |
| `outputs/contamination/cross_dataset_summary.csv` | pairwise match counts at several thresholds |
| `outputs/contamination/cross_dataset_matches.csv` | every matched pair with both file paths |
| `outputs/contamination/exact_pixel_duplicates.csv` | exact-duplicate groups |
| `outputs/contamination/within_dataset_redundancy.csv` | near-duplicate rates |
| `outputs/contamination/clinicdb_changepoint_labels.csv` | recovered vs official sequence labels |
| `outputs/splits/` | leak-safe groups, canonical folds, deduplicated keep-lists |
| `outputs/splits/training/` | the four arms × five folds used for training |
| `outputs/paper2/perimage_*.csv` | per-image metrics for all 100 runs |
| `outputs/inflation/` | paired inflation estimates and ranking shifts |
| `outputs/rescore/` | re-scored benchmark and the CVC-300 identity check |
| `outputs/pranet/` | released-checkpoint per-image metrics |
| `outputs/prior/` | position-prior per-image metrics |
| `outputs/provenance/` | CVC-ColonDB provenance record with archive URL |
| `outputs/figures/` | figures as PDF and PNG |

## Data

Images are **not redistributed**. Obtain them from the original providers:

| Dataset | Images | Source |
|---|---:|---|
| Kvasir-SEG | 1000 | https://datasets.simula.no/kvasir-seg/ |
| Kvasir-Sessile | 196 | https://datasets.simula.no/kvasir-seg/ |
| CVC-ClinicDB | 612 | https://polyp.grand-challenge.org/CVCClinicDB/ |
| CVC-ColonDB | 380 | PraNet test bundle |
| ETIS-LaribPolypDB | 196 | PraNet test bundle |
| CVC-300 | 60 | PraNet test bundle |

Use is subject to the providers' terms; Kvasir-SEG is licensed for research
and educational purposes only. Every released metric row identifies its source
image by filename so results can be joined back to the originals.

The audited copy of the PraNet test bundle has SHA-256
`b3be6da5b25679bed228dca52ae3d668dd89f329b2505c7b8961aa20dca4a90a`.

Expected layout:

```
data/Kvasir-SEG/{images,masks}/
data/sessile-main-Kvasir-SEG/{images,masks}/
data/external/<name>/{images,masks}/
```

## Reproducing

Analysis, figures and manuscript (no GPU):

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas scipy scikit-learn pillow imagehash matplotlib

.venv/bin/python scripts/01_contamination_audit.py
.venv/bin/python scripts/03_sequence_changepoint.py
.venv/bin/python scripts/06_leak_safe_groups.py
.venv/bin/python scripts/07_rescore_benchmark.py
.venv/bin/python scripts/08_provenance_record.py
.venv/bin/python scripts/09_make_splits.py
.venv/bin/python scripts/11_position_prior.py
.venv/bin/python scripts/10_analyze_inflation.py   # needs outputs/paper2/
.venv/bin/python scripts/12_figures.py
.venv/bin/python scripts/13_make_macros.py
.venv/bin/python scripts/15_make_tables.py
.venv/bin/python scripts/16_verify_manuscript.py

cd paper && latexmk -pdf main.tex
```

The 100 training runs (GPU; about six hours on one RTX 4090):

```bash
bash scripts/pod_setup.sh <host> <port>
PYTHON=/workspace/venv/bin/python bash run_paper2.sh 1   # CVC-ClinicDB arms
PYTHON=/workspace/venv/bin/python bash run_paper2.sh 2   # Kvasir-SEG arms
bash scripts/pod_sync.sh <host> <port> watch             # retrieve results
```

`run_paper2.sh` skips any run with a `summary.json`, so an interrupted job
restarts without losing work. Run the sync watcher for the duration: a pod
created without a persistent volume loses everything on termination.

The released PraNet evaluation additionally needs the PraNet repository and
its published checkpoint; see the header of `scripts/14_pranet_eval.py`.

## Negative and null results retained here

These are kept because they bound what the study can claim.

- No image is shared between Kvasir-SEG and any CVC dataset; CVC-ClinicDB and
  CVC-ColonDB share none despite common provenance; ETIS-Larib is clean
  against every other set.
- Grouping by connected components of a thresholded similarity graph
  **percolates**: by r = 0.77 all 612 CVC-ClinicDB frames collapse into one
  component. Discarded in favour of contiguous segmentation.
- Estimating the number of source sequences without ground truth was
  **unstable** (32 against a true 29 on full CVC-ClinicDB, but 27 and 25
  against a true 15 on its halves). Leak-safety is therefore certified only
  where a published sequence table exists.
- Leakage did not change the in-domain architecture ranking on CVC-ClinicDB,
  although removing the CVC-300 duplication did change it on the suite
  average. A shift in the mean and a change in ranking are distinct failures.
- Kvasir-SEG's inflation of 1.1 σ_run means the companion study's image-level
  folds were only marginally affected; its conclusions stand.

## Citation

Manuscript under preparation. See `paper/main.tex`.
