# Roadmap

This is the current implementation status of Valerie.

## Phase 0

Status: complete

Delivered:

- project framing and build plan
- agreement/disagreement and gap analysis
- initial execution strategy

## Phase 1

Status: complete

Delivered:

- Python project scaffold
- `pyproject.toml`
- package layout
- config loading with Pydantic validation
- tests and linting baseline

## Phase 2

Status: complete

Delivered:

- model backend registry
- TransformerLens model loading
- dummy backend for smoke tests
- activation extraction and serialization
- MPS verification on real models

## Phase 3

Status: complete

Delivered:

- paired framing experiment runner
- paraphrase expansion
- narrative control condition
- deterministic seeds
- manifest and sample artifact writing

## Phase 4

Status: complete

Delivered:

- linear framing probes with sklearn logistic regression
- per-layer evaluation
- task-held-out validation
- permutation baselines
- narrative-control supervised analysis
- PCA outputs
- clustering outputs
- basic layer depth plots
- GPT-2 Small and TinyStories phase-4 runs

## Phase 5 (activation patching)

Status: complete

Delivered:

- last-token residual stream patching across all layers
- matched-pair evaluation across tasks and variants
- recovery cosine and KL divergence metrics
- per-layer summary plots and pair-level CSVs
- GPT-2 Small v2 patching run (care → threat)
- Finding 002: causal verification of the framing representation

## Phase 6

Status: next

Planned focus:

- cleaner visualization polish
- clearer result presentation
- stronger results summaries
- improved analysis writeups
- possibly richer exploratory analysis if needed

## Not yet implemented

- nonlinear probes
- large-model cloud workflows
- significance testing beyond the current permutation baseline workflow
- publication-oriented reporting
- patching on components other than `resid_post` (attention patterns, MLP outputs)
- multi-token-generation patching effects

