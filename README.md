# IaCX: static structural measures for Terraform modules

This repository holds the measurement tool, test fixtures, data and analysis scripts for the article
"IaCX: A Static Structural Measurement Suite for Terraform Configurations" (Kulathunga and De Silva).

The tool parses the `*.tf` files of one Terraform module directory, builds a static source-reference graph
over its declared blocks, and reports five measures separately:

| Measure | Definition |
|---|---|
| D, dependency depth | edges on the longest directed path; not defined (reported as cyclic) when the graph has a cycle |
| P, two-step count | number of edges plus the sum over vertices of fan-in times fan-out |
| X, construct count | weighted count of `count`, `for_each`, `dynamic`, conditionals, `for` expressions, `try`/`can`, splats, `depends_on`, `lifecycle` and provisioners (weights in `EXPR_WEIGHTS`) |
| M, module calls | number of module blocks; child sources are not expanded |
| K, interface edges | edges with at least one endpoint at a module-call vertex |

The sum S = D + P + X + M + K is printed only as an exploratory comparison quantity. The five measures belong to
different measurement concepts and are not meant to be added.

The tool needs no provider credentials and never runs `terraform init` or `terraform plan`.

## Install and run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python3 iacx/iacx.py path/to/module            # human-readable report
python3 -m pytest tests/ -v                    # 16 regression tests
```

Tested with Python 3.13.16 and the pinned versions in `requirements.txt`.

## Regression tests

`tests/test_extraction.py` asserts hand-derived graphs and invariances. The worked example of the article
(Listing 1) is `examples/worked_example/base/main.tf`; the test `test_listing1_graph_and_vector_are_exact`
asserts its nine named edges and the measure vector D = 4, P = 18, X = 1.5, M = 1, K = 1, S = 25.5.
The `reordered` and `rewired` variants are the two transformations in Table 2 of the article.
The observed output of the test run is stored in `tests/observed/`.

`iacx/iacx_original.py` is the extractor as it was before the reference-resolution correction described in
Section 5.3 of the article. It is kept only so that the before-and-after comparison can be reproduced;
do not use it for measurement.

## Reproducing the study

Run every command from the repository root. Seeds are fixed, so reruns reproduce every number.

| Step | Script | Output |
|---|---|---|
| Candidate repositories | `mining/candidate_repositories.json` | 542 GitHub repositories |
| Original mining (history, labels, first measures) | `mining/mine.py` | `data/modules_corrected.csv` (19,365 directories) |
| Snapshot reconstruction, original vs corrected extractor | `mining/remine3.py` | `mining/out/*.json` (one file per repository) |
| Manifest, Table 5, exposure and distinct-commit data | `analysis/post.py` | `data/manifest.csv`, `data/remeasured_directories.csv`, `results/post/` |
| Re-measurement with v1.0.0 | `analysis/remeasure_reusable.py`, `analysis/remeasure_all.py` | `data/reusable_v1_measures.csv`, `data/all_v1_measures.csv` |
| Primary analysis (Tables 6 to 10, Section 7.4) | `analysis/analyse4.py` | `results/v4/` (results, held-out predictions, fold assignments, analysed rows) |
| All-role and unfiltered analyses (Section 7.4.1) | `analysis/analyse5.py` | `results/v5/results5.json` |
| Related-repository families (Section 7.4.2) | `analysis/families.py`, `analysis/families_rerun.py` | `data/repo_families.csv`, `results/families/` |
| Label audit sample and scoring (Section 6.3) | `analysis/fetch_commits.py`, `analysis/sample_commits.py`, `analysis/score_labels.py` | `audits/`, `results/label_audit_results.json` |
| Pre-1.0 analysis of the earlier version | `analysis/analyse3.py data/modules_corrected.csv results` | `results/results3.json` |
| Randomised property checks (Appendix A) | `analysis/randsearch.py` | console report |

`analyse4.py`, `analyse5.py`, `post.py` and `score_labels.py` run from the files in this repository. The mining and re-measurement
steps clone the repositories into `mining/clones/` (not included).

Logistic models are unregularised maximum likelihood (statsmodels `Logit`), predictors are log(1 + x), folds are ten repository-grouped
folds from scikit-learn `GroupKFold`, ROC AUC is computed on predictions pooled across folds (ties get half credit), PR AUC is average
precision, and paired intervals come from 2,000 repository-level bootstrap resamples of the fixed held-out predictions.

Extractors: `iacx/iacx.py` is version 1.0.0 and gives every reported measure. `iacx/iacx_original.py` (quoted-label defect) and
`iacx/iacx_label_corrected_pre1.py` (labels fixed, literal text still scanned) are kept only to reproduce Table 5 and the
literal-text comparison; do not use them for measurement.

`data/modules_corrected.csv` columns `commits` and `fix_commits` count `.tf` file changes per directory, not distinct commits;
`data/remeasured_directories.csv` also has distinct-commit and bot-commit counts.

## Audits

`audits/label_audit_ratings.csv` and `audits/ratings/label/{A,B}/` contain the two independent raters' judgements of 200 sampled
commits (100 keyword-flagged, 100 not). `audits/label_sample_key.json` holds the keyword flag, which was hidden from the raters.

## Licence

MIT, see `LICENSE`.
