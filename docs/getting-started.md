# Getting Started

This guide walks through the first successful run from a clean clone to exported
Chronos artifacts.

## Prerequisites

- Python `>=3.10,<3.13`
- Poetry
- Internet access for initial Hugging Face model download
- Sufficient disk space for model artifacts and training outputs

## Install

```bash
poetry lock
poetry install
```

## Step 1: Download the Chronos base model

```bash
poetry run python -m tachyon_model_downloader.download_model --config config/model_download.yaml
```

Expected output path:

- `artifacts/base-models/chronos-bolt-base`

Run state file:

- `artifacts/base-models/chronos-bolt-base/download_state.json`

## Step 2: Fine-tune and export

```bash
poetry run python -m tachyon_model_downloader.fine_tune_and_export --config config/fine_tune.yaml
```

Expected output paths:

- Predictor bundle: `artifacts/ag_predictor`
- Standalone export: `artifacts/final_safetensors`
- Training state: `artifacts/ag_predictor/training_state.json`
- Metrics report: `artifacts/ag_predictor/metrics_report.json`
- Export manifest: `artifacts/final_safetensors/export_manifest.json`

## Verify a successful run

- Confirm `artifacts/base-models/chronos-bolt-base/config.json` exists
- Confirm `artifacts/ag_predictor/training_state.json` exists
- Confirm `artifacts/final_safetensors/export_manifest.json` exists

## Common first-run issues

- Missing base model directory:
  - Ensure `training.chronos_model_path` in `config/fine_tune.yaml` points to the
    downloaded model path.
- Parquet schema mismatch:
  - Ensure train and validation parquet files contain `item_id`, `timestamp`, and `target`
    (or update config field names).
- Slow CPU training:
  - Lower `training.max_epochs` or `training.fine_tune_steps` for validation runs.
