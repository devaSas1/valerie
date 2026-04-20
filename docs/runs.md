# Runs

This file is the execution log for Valerie.

It records what has actually been run, where the artifacts live, and what each run was for. This is a lab notebook, not a polished report.

## Conventions

- Activation runs live under `data/activations/`
- Probe-analysis runs live under `data/results/`
- The activation directory name is the source of truth for the model + experiment pairing
- The probe result directory name includes both the probe config name and the activation directory it analyzed

## Phase 1-3 verification runs

### Dummy backend smoke test

Purpose:

- verify the pipeline shape without requiring model download
- test saved activation payloads and manifest generation

Artifacts:

- [`data/activations/threat-vs-care-mvp_dummy-smoke-test_20260410-222902`](../data/activations/threat-vs-care-mvp_dummy-smoke-test_20260410-222902)

Notes:

- used for automated tests and quick end-to-end verification
- not scientifically meaningful

### TinyStories-1M activation run

Purpose:

- verify the real TransformerLens path on Apple Silicon / MPS
- produce a small real-model activation set for initial phase-4 plumbing

Artifacts:

- [`data/activations/threat-vs-care-mvp_tiny-stories-1m_20260410-223305`](../data/activations/threat-vs-care-mvp_tiny-stories-1m_20260410-223305)

Notes:

- small engineering model
- useful for verifying extraction, storage, and probe wiring
- not the primary model for subsequent phase-4 analysis

## Phase 4 probe-analysis runs

### TinyStories-1M probe analysis

Purpose:

- first full phase-4 run over a real activation directory
- verify per-layer multiclass probes, permutation baselines, narrative control probes, PCA, clustering, and output structure

Source activations:

- [`data/activations/threat-vs-care-mvp_tiny-stories-1m_20260410-223305`](../data/activations/threat-vs-care-mvp_tiny-stories-1m_20260410-223305)

Results:

- [`data/results/linear-framing-probe_threat-vs-care-mvp_tiny-stories-1m_20260410-223305_20260410-234215`](../data/results/linear-framing-probe_threat-vs-care-mvp_tiny-stories-1m_20260410-223305_20260410-234215)

Notes:

- phase-4 system verification run
- sample count is too small for strong scientific claims

### GPT-2 Small activation run

Purpose:

- first intended phase-4 model run on MPS
- run the full threat/care/neutral/narrative-control activation extraction pipeline on a cached `gpt2` model

Artifacts:

- [`data/activations/threat-vs-care-mvp_gpt2-small_20260411-113704`](../data/activations/threat-vs-care-mvp_gpt2-small_20260411-113704)

Command shape:

```bash
uv run valerie-run-experiment \
  --model-config configs/models/gpt2-small.yaml \
  --experiment-config configs/experiments/threat-vs-care.yaml
```

### GPT-2 Small probe analysis

Purpose:

- produce the first real phase-4 results set on the primary model
- generate layerwise framing metrics, permutation baselines, narrative-control analysis, PCA outputs, and clustering outputs

Source activations:

- [`data/activations/threat-vs-care-mvp_gpt2-small_20260411-113704`](../data/activations/threat-vs-care-mvp_gpt2-small_20260411-113704)

Results:

- [`data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728`](../data/results/linear-framing-probe_threat-vs-care-mvp_gpt2-small_20260411-113704_20260411-113728)

Command shape:

```bash
uv run valerie-train-probes \
  --activation-dir data/activations/threat-vs-care-mvp_gpt2-small_20260411-113704 \
  --probe-config configs/probes/linear-framing.yaml
```

## GPT-2 Small v2 runs (2026-04-13)

### GPT-2 Small v2 activation run

Purpose:

- first run using `threat-vs-care-v2.yaml` — 8 tasks, diversified narrative control openers
- establishes a real leave-one-task-out evaluation (8 folds instead of 2)

Artifacts:

- [`data/activations/threat-vs-care-mvp-v2_gpt2-small_20260413-022047`](../data/activations/threat-vs-care-mvp-v2_gpt2-small_20260413-022047)

Command:

```bash
.venv/bin/valerie-run-experiment \
  --model-config configs/models/gpt2-small.yaml \
  --experiment-config configs/experiments/threat-vs-care-v2.yaml
```

### GPT-2 Small v2 probe analysis

Purpose:

- probe analysis on the v2 activation run
- first result with credible cross-validation
- produced Finding 001 (see [`findings.md`](./findings.md))

Source activations:

- [`data/activations/threat-vs-care-mvp-v2_gpt2-small_20260413-022047`](../data/activations/threat-vs-care-mvp-v2_gpt2-small_20260413-022047)

Results:

- [`data/results/linear-framing-probe_threat-vs-care-mvp-v2_gpt2-small_20260413-022047_20260413-022225`](../data/results/linear-framing-probe_threat-vs-care-mvp-v2_gpt2-small_20260413-022047_20260413-022225)

Command:

```bash
.venv/bin/python -m valerie.probes.trainer \
  --activation-dir data/activations/threat-vs-care-mvp-v2_gpt2-small_20260413-022047 \
  --probe-config configs/probes/linear-framing.yaml
```

Notes:

- 100% multiclass accuracy at layers 8–10, 100% narrative control at layers 6–11
- token length delta uniform at 7 tokens across all 8 tasks, no warnings

### GPT-2 Small v2 activation patching

Purpose:

- causal verification of the framing representation identified by the probe
- patch the last-token `resid_post` from a care run into a threat run at each layer and measure how much of the care-vs-threat logit direction is recovered
- produced Finding 002 (see [`findings.md`](./findings.md))

Source activations:

- [`data/activations/threat-vs-care-mvp-v2_gpt2-small_20260413-022047`](../data/activations/threat-vs-care-mvp-v2_gpt2-small_20260413-022047)

Results:

- [`data/patching/patching_threat-vs-care-mvp-v2_gpt2-small_20260413-022047_20260413-105440`](../data/patching/patching_threat-vs-care-mvp-v2_gpt2-small_20260413-022047_20260413-105440)

Command:

```bash
.venv/bin/python -m valerie.patching.runner \
  --activation-dir data/activations/threat-vs-care-mvp-v2_gpt2-small_20260413-022047 \
  --model-config configs/models/gpt2-small.yaml \
  --clean-condition care \
  --corrupted-condition threat \
  --component resid_post
```

Notes:

- 24 matched pairs (8 tasks × 3 variants)
- recovery cosine 0.954 at layers 9–10, 1.000 at layer 11
- patching only last-token position to control for 7-token framing wrapper length delta

## What to run next

The next run should use `threat-vs-care-v2.yaml`. That config has 8 tasks, which makes the leave-one-task-out evaluation meaningful (7 train / 1 test per fold instead of 1/1).

Standard run shape:

```bash
.venv/bin/python -m valerie.experiments.runner \
  --model-config configs/models/gpt2-small.yaml \
  --experiment-config configs/experiments/threat-vs-care-v2.yaml
```

Then probe it:

```bash
.venv/bin/python -m valerie.probes.trainer \
  --activation-dir data/activations/<dir> \
  --probe-config configs/probes/linear-framing.yaml
```

Add the resulting pair of directories to this file when done.

