# Acquisition-admissibility audit for machinery diagnosis

This repository supports the manuscript **Before model accuracy: Acquisition-admissibility gates expose specimen-identity shortcuts in machinery diagnosis**.

Release v2.0.1 preserves the frozen unified V2 numerical evidence and clarifies the mixed-licence boundary; V2 denotes the computational evidence freeze used by the verification scripts. No scientific result changed from v2.0.0.

The authoritative logistic evidence is the unified V2 record under `unified_v2/`. It regenerates the primary split comparison and all logistic controls in one pinned environment from the same frozen 1,200-row feature table. The source audits and frozen feature table remain under `primary/`; superseded pre-V2 logistic summaries are intentionally omitted so the unified V2 release is the authoritative numerical record for the manuscript. Historical method scripts are retained under `primary/historical_v1_code/` for provenance only.

## One-command verification

From a clean extraction with Python 3.10 or later:

```text
python run_all.py
```

This standard-library verification checks repository hashes, the original public inventory, the complete 90-fit CNN factorial, all 21,600 CNN prediction rows, and the unified V2 evidence.

The V2-only verification is:

```text
python run_unified_v2.py
```

It verifies the frozen feature SHA-256, 55 baseline folds, 13,200 out-of-fold predictions, 1,200 pseudolabel evaluations, 45 class-conditional identity folds, the 200-assignment fault-association sensitivity, all 552 fault-orthogonal mappings, 375 unique triplet fits and 43,200 partition-modality rows representing all 14,400 admissible bearing partitions.

## Re-running the unified logistic analysis

Install the exact dependencies in `unified_v2/code/requirements_v2_exact.txt`, then run:

```text
python unified_v2/code/run_v2_portable.py --jobs 8
```

The released evidence is already frozen and verified; retraining is optional. Raw Paderborn signals are not redistributed. Rebuilding the feature table from raw signals requires downloading the source archives directly from the Paderborn University Bearing DataCenter and accepting its source terms.

## Claim boundary

The results support a specimen-identity shortcut mechanism and split-unit sensitivity within the declared 15-bearing, one-rig Paderborn inventory. They do not estimate unrestricted industrial deployment performance, identify the physical origin of the specimen signature, or prove that every diagnostic architecture has the same gap. The 200 assignments and 552 mappings are finite challenge spaces rather than independent physical experiments; the 14,400-partition distribution is finite design sensitivity, not a confidence interval.

## Licences and attribution

This is a mixed-licence release. Author-created code is released under the
MIT License. Paderborn-derived caches, features, predictions and evidence
remain subject to CC BY-NC 4.0 and source attribution. The file-level map is
defined in `LICENSES.md`; source details are in `DATA_PROVENANCE.md`.

