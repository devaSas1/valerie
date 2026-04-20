# Results Ledger

This file is a compact index of current result artifacts.

It is not an interpretation document. It is a lookup table for where the outputs live.

## TinyStories-1M phase-4 results

Run directory:

- [`data/results/linear-framing-probe_threat-vs-care-mvp_tiny-stories-1m_20260410-223305_20260410-234215`](../data/results/linear-framing-probe_threat-vs-care-mvp_tiny-stories-1m_20260410-223305_20260410-234215)

Important files:

- [`manifest.json`](../data/results/linear-framing-probe_threat-vs-care-mvp_tiny-stories-1m_20260410-223305_20260410-234215/manifest.json)
- [`multiclass_metrics.csv`](../data/results/linear-framing-probe_threat-vs-care-mvp_tiny-stories-1m_20260410-223305_20260410-234215/metrics/multiclass_metrics.csv)
- [`narrative_control_metrics.csv`](../data/results/linear-framing-probe_threat-vs-care-mvp_tiny-stories-1m_20260410-223305_20260410-234215/metrics/narrative_control_metrics.csv)
- [`clustering_metrics.csv`](../data/results/linear-framing-probe_threat-vs-care-mvp_tiny-stories-1m_20260410-223305_20260410-234215/metrics/clustering_metrics.csv)
- [`multiclass_accuracy_by_layer.png`](../data/results/linear-framing-probe_threat-vs-care-mvp_tiny-stories-1m_20260410-223305_20260410-234215/plots/multiclass_accuracy_by_layer.png)

## GPT-2 Small phase-4 results

Run directory:

- [`data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728`](../data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728)

Important files:

- [`manifest.json`](../data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728/manifest.json)
- [`multiclass_metrics.csv`](../data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728/metrics/multiclass_metrics.csv)
- [`multiclass_predictions.csv`](../data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728/metrics/multiclass_predictions.csv)
- [`narrative_control_metrics.csv`](../data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728/metrics/narrative_control_metrics.csv)
- [`clustering_metrics.csv`](../data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728/metrics/clustering_metrics.csv)
- [`difference_pca_summary.json`](../data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728/metrics/difference_pca_summary.json)
- [`multiclass_accuracy_by_layer.png`](../data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728/plots/multiclass_accuracy_by_layer.png)
- [`narrative_control_accuracy_by_layer.png`](../data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728/plots/narrative_control_accuracy_by_layer.png)
- [`activation_pca/`](../data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728/plots/activation_pca)
- [`tables/activation_pca/`](../data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728/tables/activation_pca)
- [`tables/difference_pca/`](../data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728/tables/difference_pca)
- [`tables/cluster_assignments.csv`](../data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728/tables/cluster_assignments.csv)

## GPT-2 Small v2 results (8 tasks, first real cross-validation)

Run directory:

- [`data/results/linear-framing-probe_threat-vs-care-mvp-v2_gpt2-small_20260413-022047_20260413-022225`](../data/results/linear-framing-probe_threat-vs-care-mvp-v2_gpt2-small_20260413-022047_20260413-022225)

Important files:

- [`manifest.json`](../data/results/linear-framing-probe_threat-vs-care-mvp-v2_gpt2-small_20260413-022047_20260413-022225/manifest.json)
- [`multiclass_metrics.csv`](../data/results/linear-framing-probe_threat-vs-care-mvp-v2_gpt2-small_20260413-022047_20260413-022225/metrics/multiclass_metrics.csv)
- [`narrative_control_metrics.csv`](../data/results/linear-framing-probe_threat-vs-care-mvp-v2_gpt2-small_20260413-022047_20260413-022225/metrics/narrative_control_metrics.csv)

Key results:

- Multiclass best accuracy: **100%** at layers 8–10
- Narrative control best accuracy: **100%** at layers 6–11
- 8-task leave-one-task-out cross-validation — first credible evaluation
- See [`findings.md`](./findings.md) for full interpretation

## GPT-2 Small v2 patching results (causal verification)

Run directory:

- [`data/patching/patching_threat-vs-care-mvp-v2_gpt2-small_20260413-022047_20260413-105440`](../data/patching/patching_threat-vs-care-mvp-v2_gpt2-small_20260413-022047_20260413-105440)

Important files:

- [`manifest.json`](../data/patching/patching_threat-vs-care-mvp-v2_gpt2-small_20260413-022047_20260413-105440/manifest.json)
- [`patching_summary.csv`](../data/patching/patching_threat-vs-care-mvp-v2_gpt2-small_20260413-022047_20260413-105440/metrics/patching_summary.csv)
- [`patching_pairs.csv`](../data/patching/patching_threat-vs-care-mvp-v2_gpt2-small_20260413-022047_20260413-105440/metrics/patching_pairs.csv)
- [`recovery_by_layer.png`](../data/patching/patching_threat-vs-care-mvp-v2_gpt2-small_20260413-022047_20260413-105440/plots/recovery_by_layer.png)

Key results:

- Activation patching on last-token `resid_post`, care → threat direction
- 24 matched pairs (8 tasks × 3 variants)
- Recovery cosine **0.954** at layers 9–10 (mean across pairs), **1.000** at layer 11
- First layer with positive minimum pair recovery: layer 6 (matches probe onset)
- Establishes that the framing representation is causally load-bearing, not passive
- See [`findings.md`](./findings.md) Finding 002 for full interpretation

## How to use this ledger

When a new empirical run is completed:

1. add the result directory here
2. add the corresponding activation directory to `docs/runs.md`
3. keep the list concise and path-first

