# chronos-finetuning

A reproducible, config-driven pipeline that produces baseline vs fine-tuned
probabilistic forecast metrics for Chronos time-series models.

The project is intentionally a **metrics producer**, not an interpreter.
Win/lose, pass/fail, and recommendation logic are out of scope; downstream
analysis hooks (optionally Codex) handle interpretation.

## Project promise

Given one parquet dataset and one offline safetensors base model directory,
produce a baseline-vs-finetuned raw metric comparison and an analysis payload
that an external LLM can interpret.

## Core constraints

- Parquet-only input data.
- Safetensors-only offline model directories. No `.pkl`, `.pickle`, or `.bin`
  artifacts are loaded at inference time.
- All paths and credentials live in YAML or environment variables. No
  hardcoded paths or secrets.
- Schema role resolution is deterministic and documented.

## Requirements

- Python 3.10 to 3.12
- Poetry

## Setup

```bash
poetry lock
poetry install
```

## Project structure

- `src/tachyon_model_downloader/` - package source
- `config/` - YAML configuration files
- `scripts/` - helper shell scripts
- `docs/` - workflow and configuration reference

## Workflow

See [`docs/workflow.md`](docs/workflow.md) for the end-to-end stages and
[`docs/configuration.md`](docs/configuration.md) for the YAML reference.
All runnable Python and script commands in project docs are intended to be
invoked via `poetry run`.

## Notes

The package directory name (`tachyon_model_downloader`) is retained for
backward compatibility but the project now spans download, fine-tuning,
inference, comparison, and analysis hooks.
