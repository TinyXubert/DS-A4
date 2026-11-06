# DS-A4: Model Comparison and Calibration Lab

**Author:** xuhongbo · **Student ID:** 202618018629048

Compare Logistic Regression, Random Forest and Histogram Gradient Boosting on the UCI Wisconsin Diagnostic Breast Cancer dataset. All model families share training folds and a fixed fit-count budget. Separate partitions fit models, calibrate probabilities, choose a hypothetical cost/capacity policy and evaluate it. The notebook retains paired uncertainty, morphology slices, failures and a ranking-preserving calibration counterexample. This is a statistical teaching experiment, not a clinical decision system.

## Read the result

- [A4.pdf](A4.pdf): code, executed results and the author's independent verification/reflection.
- [A4.ipynb](A4.ipynb): executable notebook with saved outputs; starts with the actual-result Abstract.
- [METHOD_NOTE.md](METHOD_NOTE.md): method, assumptions, metrics and policy limitations.
- AI assistance is disclosed in the notebook; the author's supplied independent verification/reflection is included at its end and in the PDF.

The report uses `results/20261010T104325122644Z/`. On its 86 test records, calibrated Logistic has Brier score 0.024833221. The policy reviews 21 records and misses 11 of 32 positives under the hypothetical 30% review capacity. High discrimination therefore does not eliminate costly decision errors. Full model comparisons, uncertainty and negative calibration results remain in the notebook and result tables.

## Reproduce

Use Python 3.10 and run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python execute_notebook.py
python export_report.py
```

Alternatively select that environment in Jupyter and run `A4.ipynb` from top to bottom. `analysis.py` contains the executable analysis; `protocol.json` fixes the design and seeds. Each run creates a new `results/<UTC timestamp>/` with raw predictions, fold scores, bootstrap records, figures, completion metadata and a full dependency lock. Previous runs, negative comparisons and counterexamples are retained. Execution failures, if any, are saved by `execute_notebook.py` rather than discarded.

To export a new PDF:

```bash
python -m pip install -r requirements-export.txt
python -m playwright install chromium --only-shell
python export_report.py --pdf
python verify_results.py
```

The exporter preserves the author-provided reflection already saved in the notebook. An optional local AI log can update that section, but it is not required to export a fresh clone. The exporter also creates optional local HTML/Markdown exports, figures and duplicate summaries; these generated copies are ignored by Git. The verification script creates a local `EXECUTION_CHECK.md` and independently recomputes the Brier score and capacity check. Export requires Chromium; verifying the PDF requires `pdftotext`, `pdfinfo` and `pdfimages` from Poppler. Automated checks do not substitute for student verification or independent clinical validation.

## Data and provenance

The archived [UCI dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) is attributed to Wolberg, Mangasarian, Street and Street (1993), DOI [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B), **CC BY 4.0**. It comprises existing anonymized numerical image measurements; IDs are excluded from predictive inputs. No patient recruitment or new personal records are claimed. Preserve upstream attribution and data terms; the repository's existing `LICENSE` does not replace the dataset license.

`data/raw/manifest.json` records URLs, download time and SHA-256 for `wdbc.data`, `wdbc.names` and the original `source.zip`; all three are retained because the analysis verifies their hashes. Data SHA-256 for `wdbc.data` is `d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a`.

## Evidence and repository scope

`results/` retains all three completed runs, including their raw tables, discussion/Abstract, figures, environment locks and protocol copies. The seven notebook code cells were executed in order with no error outputs. Submission preparation on 2026-11-06 rechecked the saved PDF, source/protocol identity, direct Brier calculation, capacity rule and repeated-run summary consistency; the experiments were not rerun to obtain different results.

The repository contains the notebook, PDF, four Python scripts, dependency files, frozen protocol, original data, all result evidence, README and method note. The standalone AI log remains local; notebook/PDF disclosure and reflection are retained. Local environments, caches, duplicate exports and platform-upload reminders are excluded. Course-platform registration and any Google Drive upload are separate from this GitHub repository.
