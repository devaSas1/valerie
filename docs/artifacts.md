# Artifacts

Each experiment run writes a timestamped artifact directory.

## Output directory structure

Example:

```text
data/activations/
  threat-vs-care-mvp_tiny-stories-1m_20260410-162340/
    manifest.json
    resolved_model_config.json
    resolved_experiment_config.json
    samples/
      summarize_paragraph__neutral__variant_0.pt
      summarize_paragraph__threat__variant_0.pt
      ...
```

## `manifest.json`

The manifest is the run-level index. It contains:

- experiment metadata
- model metadata
- selected device
- number of model layers
- deterministic seed
- resolved extraction settings
- per-task prompt length summaries
- warnings
- sample inventory

Each sample entry includes:

- `sample_id`
- `task_id`
- `condition_name`
- `variant_index`
- relative file path

## Sample payloads

Each `.pt` file is a serialized PyTorch dictionary with these keys:

- `prompt`: rendered prompt text
- `token_ids`: token tensor for the prompt
- `activations`: mapping from standardized activation keys to tensors
- `logits`: optional logit tensor
- `metadata`: sample metadata added by the runner

## Activation key format

Activation tensors are stored under keys like:

- `resid_post.layer_0`
- `resid_post.layer_7`
- `mlp_post.layer_3`

The exact keys depend on:

- requested components
- requested layers
- model depth

## Token-position behavior

Position handling is controlled by the experiment config:

- `all`: save all positions
- `last`: save only the last sequence position
- `index`: save one explicit position
- `mean_pool`: average across sequence positions

For attention patterns, the selection logic treats the query-position axis specially so the output shape stays meaningful.

## Inspecting a payload

Example:

```bash
uv run python -c '
import torch
p = torch.load("data/activations/.../samples/example.pt")
print(p["metadata"])
print(sorted(p["activations"].keys())[:5])
print({k: tuple(v.shape) for k, v in list(p["activations"].items())[:2]})
'
```

## Why the artifacts look like this

The current format is optimized for:

- easy inspection
- simple downstream probe training
- explicit run metadata
- compatibility with notebooks and scripts

It is not optimized yet for massive-scale storage efficiency. If artifact volume becomes a problem, we can revisit formats in a later phase.

