# Config Reference

Valerie is config-driven. The default rule is: if a researcher should be able to change it between runs, it should probably live in YAML.

## Model configs

Directory: `configs/models/`

Example:

```yaml
name: tiny-stories-1m
backend: transformer_lens
model_name: tiny-stories-1M
device_preference:
  - mps
  - cuda
  - cpu
dtype: float32
default_prepend_bos: true
trust_remote_code: false
n_ctx: 512
quantization:
  mode: none
seed: 0
```

### Fields

- `name`: human-readable label for artifacts
- `backend`: `transformer_lens` or `dummy`
- `model_name`: backend-specific model identifier
- `device_preference`: priority-ordered list of devices to try
- `dtype`: tensor dtype for model loading
- `default_prepend_bos`: whether prompts get a BOS token by default
- `trust_remote_code`: passed through to the backend when supported
- `n_ctx`: reserved in the schema, but not currently passed through to TransformerLens loading
- `first_n_layers`: optional truncation for supported backends
- `quantization.mode`: `none`, `4bit`, or `8bit`
- `cache_dir`: optional Hugging Face cache path
- `dummy_d_model`: dummy backend only
- `dummy_n_layers`: dummy backend only
- `seed`: reserved for future backend-level deterministic behavior

### Notes

- `n_ctx` is present in the schema for future use, but the current TransformerLens adapter does not pass it through because it broke some Hugging Face-backed model classes.
- Quantization flags are backend-dependent. They are wired for the TransformerLens path, but not every model/backend combination will support every mode.

## Experiment configs

Directory: `configs/experiments/`

Example sections:

```yaml
extraction:
  components:
    - resid_post
    - mlp_post
  layers: all
  position:
    strategy: last
  save_logits: true
```

```yaml
controls:
  require_matched_variant_counts: true
  warn_on_char_length_delta_over: 40
  deterministic_seed: 0
```

```yaml
conditions:
  threat:
    description: Self-directed threat framing.
    target: self
    variants:
      - "Complete the task correctly. A wrong answer will count as a serious failure.\n\nTask:\n{task}"
```

### Top-level fields

- `name`: experiment label used in output directories
- `description`: human-readable description
- `extraction`: what internal tensors to capture
- `controls`: experiment-level reproducibility and guardrails
- `conditions`: named framing conditions
- `tasks`: the base task prompts to render under each condition

### Extraction fields

- `components`: list of supported activation components
- `layers`: `all` or an explicit list of integers
- `position.strategy`: `all`, `last`, `index`, or `mean_pool`
- `position.index`: required when strategy is `index`
- `save_logits`: whether to include logits in each payload

### Condition fields

- `description`: plain-English condition description
- `target`: `self`, `narrative`, or `neutral`
- `variants`: paraphrase list for the condition

### Task fields

- `id`: stable task identifier used in sample ids
- `prompt`: the raw task text
- `metadata`: optional key/value metadata

### Validation rules

Current schema validation enforces:

- experiments must include a `neutral` condition
- experiments must include at least one task
- each condition must have at least one variant
- if `require_matched_variant_counts` is true, all conditions must have the same number of variants
- `position.index` must be set when `position.strategy` is `index`

## Probe configs

Directory: `configs/probes/`

Probe configs are scaffolded but not yet used by the codebase. The current example is laying groundwork for phase 4.

