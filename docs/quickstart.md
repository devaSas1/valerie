# Quickstart

This guide gets Valerie running locally and walks through the first framing experiment.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- Apple Silicon with MPS or an NVIDIA GPU with CUDA is recommended
- CPU-only runs are possible for the dummy backend and very small models

## Install `uv`

If `uv` is not already installed:

```bash
python3 -m pip install --user uv
```

On this machine, the binary may land in:

```bash
/Users/devasasikumar/Library/Python/3.9/bin
```

If `uv` still is not found after installation, add that directory to your `PATH`:

```bash
export PATH="/Users/devasasikumar/Library/Python/3.9/bin:$PATH"
```

Or run `uv` directly by absolute path:

```bash
/Users/devasasikumar/Library/Python/3.9/bin/uv --version
```

## Install the project

From the repository root:

```bash
uv sync --extra dev
```

If `uv` is installed but not on your `PATH`, use:

```bash
/Users/devasasikumar/Library/Python/3.9/bin/uv sync --extra dev
```

This creates `.venv/` and installs runtime plus development dependencies.

## Run the first smoke test

The dummy backend is the fastest way to verify the pipeline shape without downloading a model:

```bash
uv run valerie-run-experiment \
  --model-config configs/models/dummy.yaml \
  --experiment-config configs/experiments/threat-vs-care.yaml
```

This should print an output directory under `data/activations/` or your chosen output path.

## Run a real TransformerLens model

The repository includes a small example config for `tiny-stories-1M`:

```bash
uv run valerie-run-experiment \
  --model-config configs/models/tiny-stories-1m.yaml \
  --experiment-config configs/experiments/threat-vs-care.yaml
```

This path does real model loading and activation extraction through TransformerLens.

## What the command does

The experiment runner:

- loads the model config
- loads the experiment config
- selects a device from the configured preference order
- applies deterministic seeds
- builds all task/condition/variant prompt combinations
- runs each prompt through the model
- extracts the requested activations
- saves one `.pt` payload per sample
- writes a `manifest.json` plus resolved configs

## Example output

You should get a directory shaped roughly like this:

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

See [`artifacts.md`](./artifacts.md) for the payload contents.

## Development commands

Run tests:

```bash
uv run pytest
```

Run lint:

```bash
uv run ruff check .
```

## Known caveat

TransformerLens-backed models may still make Hugging Face metadata requests even after weights are cached. In practice this means:

- a normal online run works
- a fully sandboxed or DNS-blocked run may fail unless we add an explicit offline/cache mode later

The dummy backend is available for offline pipeline development and tests.
