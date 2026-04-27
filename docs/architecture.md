# Architecture and Design

Chronos Fine-Tuning is organized as a small, modular CLI package with strict
schema validation and config-driven execution.

## Module map

- `src/tachyon_model_downloader/schemas.py`
  - Pydantic models for model download and state schema validation.
- `src/tachyon_model_downloader/download_model.py`
  - Download pipeline entry point and idempotent snapshot acquisition.
- `src/tachyon_model_downloader/fine_tune_and_export.py`
  - Data loading, AutoGluon fine-tuning, export, and metrics reporting.

## High-level flow

1. Input YAML is parsed and validated via Pydantic models.
2. Paths are resolved relative to project root unless absolute.
3. Workflows execute in deterministic output locations.
4. JSON state artifacts are generated after each major stage.

## Design principles

- Config-driven runtime behavior
  - Model IDs, file paths, and training parameters are controlled in YAML.
- Idempotent download stage
  - Existing model snapshots are reused unless forced download is enabled.
- Structured outputs
  - Each run writes machine-readable manifests and state reports.
- Separation of concerns
  - Download and fine-tune/export concerns are separated into distinct modules.

## Generated artifacts

- Download stage:
  - `download_state.json`
- Fine-tune stage:
  - `training_state.json`
  - `metrics_report.json`
- Export stage:
  - `export_manifest.json`
  - exported model files such as `model.safetensors` when available

## Extending the project

- Add new models by changing `model.repo_id` and `training.chronos_model_path`.
- Add new datasets by updating `data.*` paths and column mapping.
- Introduce additional run reports by adding new Pydantic report models and
  serializing them in the workflow entry points.
