# Valerie Docs

These docs are the operational companion to the top-level `README.md`.

- The repository `README.md` is the project's public framing and philosophical motivation.
- The `docs/` directory is the engineering and research-operations layer.
- If the README is the manifesto, these docs are the field manual.

## Start here

- [`quickstart.md`](./quickstart.md): install the environment and run the first experiment
- [`architecture.md`](./architecture.md): package layout, layer boundaries, and data flow
- [`configs.md`](./configs.md): model and experiment config reference
- [`experiments.md`](./experiments.md): framing conditions, controls, and current methodology
- [`artifacts.md`](./artifacts.md): what gets saved to disk and how to inspect it
- [`runs.md`](./runs.md): execution log for completed activation and probe runs
- [`results.md`](./results.md): ledger of current result artifact directories
- [`probes.md`](./probes.md): how phase-4 probe training and evaluation currently work
- [`methodology.md`](./methodology.md): operational research design and current guardrails
- [`roadmap.md`](./roadmap.md): project status by phase
- [`findings.md`](./findings.md): timestamped record of confirmed empirical findings

## Current scope

The current codebase covers phases 1-4 of the initial build:

- project scaffold and packaging
- validated YAML configs
- model loading backends
- activation extraction
- paired framing experiment runner
- linear framing probes
- task-held-out evaluation
- permutation baselines
- narrative-control probe analysis
- PCA and clustering outputs
- baseline tests and linting

The current codebase does not yet include:

- activation patching
- nonlinear probes
- cloud execution workflows
- publication-oriented result presentation

Those come after the current phase-4 milestone.
