# Experiment Design

This document covers the current MVP framing experiment and the methodological controls already built into the config structure.

## Current MVP question

Can we detect framing-correlated internal representations in the model when the same underlying task is presented under different motivational framings?

The current operational language is intentionally narrower than emotion language. The pipeline works with:

- `condition_name`
- `condition_target`
- framing variants

It does not currently claim that a decodable signal is an emotion.

## Experiment configs

There are currently two experiment configs:

- [`configs/experiments/threat-vs-care.yaml`](../configs/experiments/threat-vs-care.yaml): original MVP with 2 tasks and 3 variants per condition. Used for initial infrastructure verification and the first GPT-2 Small runs.
- [`configs/experiments/threat-vs-care-v2.yaml`](../configs/experiments/threat-vs-care-v2.yaml): expanded config with 8 tasks and diversified narrative control openers. This is the recommended config for any new runs.

## Current conditions

Both configs include four conditions:

- `neutral`: task prompt with no explicit motivational framing
- `threat`: self-directed threat framing
- `care`: self-directed care framing
- `narrative_threat_control`: threat language appears, but it is about a character in a story rather than directed at the model

That last condition is especially important. It helps separate:

- lexical threat content
- self-directed motivational framing

The v2 config uses varied openers for the narrative control variants ("A character in a story...", "In this fictional scenario...", "Setting: someone in a story...") to avoid the original "Story context:" prefix acting as a trivial structural cue for the probe.

## Current controls

The default experiment config already includes several controls:

- Multiple paraphrases per condition
- Matched variant counts across conditions
- Length-delta warnings for prompt matching
- Deterministic seeding
- Neutral baseline
- Narrative threat control

These controls are not perfect, but they are good enough for an infrastructure-first MVP.

## Why paraphrases matter

Without paraphrases, a probe can win by memorizing specific wording patterns rather than learning anything about framing at the representation level.

Paraphrase families push the task toward detecting more stable internal structure.

## Why prompt-length matching matters

Large prompt-length differences are a confound:

- they change token counts
- they change positional structure
- they can trivially shift activations

The runner currently computes per-task character-length deltas and records warnings in `manifest.json` when the configured threshold is exceeded.

This is only a guardrail, not a guarantee. Character length is a crude proxy for token-level matching, but it is still useful.

## Deterministic settings

The runner currently sets seeds for:

- Python `random`
- NumPy, when installed
- PyTorch, when installed

This helps reduce accidental run-to-run variability while the project is still at the infrastructure stage.

## What phase 4 will add

The next stage should train framing probes on the saved activation tensors and report:

- train/test splits that hold out task contexts
- binary threat-vs-care classification
- control comparisons such as threat vs narrative threat
- layer-by-layer separability

## What phase 5 will add

Phase 5 should include both supervised and unsupervised analysis:

- layer-by-layer framing signal depth profiles
- PCA or related low-dimensional projections
- clustering on activation differences
- exploratory structure discovery beyond predefined labels

That balance matters. If we only ask whether predefined labels are present, we may miss structure the model actually uses.

