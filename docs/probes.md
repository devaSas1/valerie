# Probes

This document describes the current phase-4 probe workflow.

## Current objective

The current supervised objective is framing classification, not emotion classification.

The default probe config uses:

- `threat`
- `care`
- `neutral`

as the multiclass labels, and separately compares:

- `threat`
- `narrative_threat_control`

for the strongest confound-control check.

## Current implementation

Code:

- [`src/valerie/probes/dataset.py`](../src/valerie/probes/dataset.py)
- [`src/valerie/probes/linear.py`](../src/valerie/probes/linear.py)
- [`src/valerie/probes/trainer.py`](../src/valerie/probes/trainer.py)

Config:

- [`configs/probes/linear-framing.yaml`](../configs/probes/linear-framing.yaml)

## Data loading

The probe pipeline reads a saved activation run and:

- loads `manifest.json`
- loads resolved model and experiment configs
- loads each `.pt` sample payload
- extracts one configured activation component, currently `resid_post`
- flattens each layer activation into a feature vector

This produces one feature matrix per layer.

## Supervised probe

The default probe is a scikit-learn pipeline:

- `StandardScaler`
- `LogisticRegression`

This was chosen because:

- it is simple
- it is easy to audit
- it matches the “linear probe first” strategy

## Evaluation protocol

The current evaluation uses leave-one-task-out cross-validation.

That means:

- the model trains on one task family
- the model tests on the held-out task family
- the split key is `task_id`

This is a stronger test than random sample splitting because it checks whether the framing signal generalizes across task contexts instead of memorizing prompt specifics.

## Reported metrics

For each layer, the multiclass and narrative-control analyses report:

- accuracy
- macro F1
- macro one-vs-rest AUROC

## Permutation baseline

Each layer also gets a permutation baseline.

Current behavior:

- labels are shuffled within task groups
- the same held-out evaluation protocol is rerun
- this happens `num_permutations` times from the probe config
- the mean and standard deviation are saved per metric

This provides a chance-level reference under the same dataset geometry.

## Narrative control analysis

The pipeline runs a second supervised analysis on:

- `threat`
- `narrative_threat_control`

This is important because it tests whether self-directed threat framing is distinguishable from narrative threat language.

The results are written separately from the multiclass framing probe.

## Unsupervised outputs

Phase 4 also writes:

- activation PCA per layer
- threat-care paired-difference PCA summaries
- clustering metrics and cluster assignments

These are exploratory outputs, not definitive analyses.

## Output structure

A probe run writes:

- `manifest.json`
- metrics CSV and JSON files
- saved per-layer probe models
- layerwise accuracy plots
- PCA tables and scatter plots
- clustering metrics and assignments

See [`results.md`](./results.md) for the specific current runs.

