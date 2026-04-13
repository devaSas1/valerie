# Architecture

Valerie is organized as a small research platform with clear layer boundaries. The goal is not enterprise architecture. The goal is to keep experiments legible, configurable, and easy to extend.

## Design principles

- Config-driven first
- Thin abstractions over real tensors
- Operational language in code and configs
- Reproducible artifacts
- Minimal magic

## Package layout

```text
src/valerie/
  config.py
  models/
  extraction/
  experiments/
  probes/
  analysis/
configs/
  models/
  experiments/
  probes/
scripts/
tests/
docs/
```

## Layer overview

### 1. Config loading

File: `src/valerie/config.py`

Responsibilities:

- parse YAML
- validate schema with Pydantic
- reject unknown fields
- provide strongly typed config objects to the rest of the code

Why it matters:

- experiments should fail early on bad config
- we want reproducibility through explicit config, not hidden code defaults

### 2. Model management

Files:

- `src/valerie/models/loader.py`
- `src/valerie/models/registry.py`

Responsibilities:

- select a device (`mps`, `cuda`, `cpu`)
- instantiate a backend
- expose a common `run_with_cache(prompt)` interface

Current backends:

- `transformer_lens`: real model loading and activation caching
- `dummy`: deterministic fake runtime for tests and smoke checks

The dummy backend exists to keep the pipeline testable even when model downloads are unavailable.

### 3. Activation extraction

Files:

- `src/valerie/extraction/hooks.py`
- `src/valerie/extraction/activations.py`

Responsibilities:

- map Valerie component names to backend cache keys
- resolve which layers to extract
- select token positions (`all`, `last`, `index`, `mean_pool`)
- standardize extracted tensors into a consistent payload

Current supported components:

- `resid_pre`
- `resid_mid`
- `resid_post`
- `mlp_pre`
- `mlp_post`
- `attn_pattern`
- `head_result`

### 4. Experiment management

Files:

- `src/valerie/experiments/framings.py`
- `src/valerie/experiments/runner.py`

Responsibilities:

- render framed prompts from task templates
- expand all condition/variant combinations
- seed deterministic execution
- run all samples
- save payloads and a manifest
- warn when prompt-length matching is poor

### 5. Probes and analysis

Files:

- `src/valerie/probes/*`
- `src/valerie/analysis/*`

These are currently stubs. They are reserved for the next phases:

- linear framing probes
- layerwise separability analysis
- unsupervised exploratory analysis
- visualization
- significance testing

## Data flow

At a high level:

```text
YAML config
  -> validated config objects
  -> model backend load
  -> framed prompt expansion
  -> forward pass with cache
  -> activation extraction
  -> serialized sample payloads
  -> manifest + resolved configs
```

## Why the architecture looks like this

- The config layer isolates experiment definition from code changes.
- The model layer isolates backend differences from the experiment runner.
- The extraction layer keeps cache-key logic out of experiments.
- The experiment layer owns reproducibility and artifact structure.
- The analysis layer stays separate so plotting and statistics do not leak into inference code.

That separation should keep phase 4 and phase 5 work straightforward.

