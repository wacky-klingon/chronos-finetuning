# Chronos Fine-Tuning

Chronos Fine-Tuning helps teams bring Amazon Chronos Bolt forecasting models into
private workflows: download once, fine-tune on internal time-series data, and
export offline model artifacts for local deployment.

The project is configuration-first and designed for repeatable execution through
YAML files, state metadata, and deterministic output paths.

## Why This Project

- Build Chronos-based forecasting pipelines without managed cloud lock-in
- Fine-tune on proprietary data while keeping model artifacts in your environment
- Keep model lifecycle steps reproducible through config and state tracking
- Move from base model download to offline export with a focused CLI workflow

## Core Capabilities

- Download Hugging Face Chronos model snapshots to a local artifact directory
- Skip redundant downloads when artifacts already exist (idempotent behavior)
- Fine-tune Chronos through AutoGluon TimeSeries with YAML-configured parameters
- Export a standalone `safetensors` package for local inference and distribution
- Write machine-readable run metadata (`download_state`, `training_state`, metrics)

## Quick Start

### 1) Install

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Bash:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Download the base model

```bash
python -m tachyon_model_downloader.download_model --config config/model_download.yaml
```

### 3) Fine-tune and export

```bash
python -m tachyon_model_downloader.fine_tune_and_export
```

## Check-in Hygiene

Install project and development tooling with Poetry:

```bash
poetry install --with dev
```

Run local quality checks:

```bash
poetry run ruff check .
poetry run ruff format .
poetry run mypy src
poetry run bandit -r src
```

Enable automated checks at commit time:

```bash
poetry run pre-commit install
```

Run all configured hooks on demand:

```bash
poetry run pre-commit run --all-files
```

## Typical Outputs

- Base model snapshot: `artifacts/base-models/chronos-bolt-base`
- Download run state: `artifacts/base-models/chronos-bolt-base/download_state.json`
- Fine-tuned predictor bundle: `artifacts/ag_predictor`
- Training run state: `artifacts/ag_predictor/training_state.json`
- Training metrics report: `artifacts/ag_predictor/metrics_report.json`
- Standalone export bundle: `artifacts/final_safetensors`
- Export manifest: `artifacts/final_safetensors/export_manifest.json`

## Documentation

Detailed technical material is split into focused docs:

- [Getting Started](docs/getting-started.md)
- [Pipeline Workflow](docs/workflow.md)
- [Configuration Reference](docs/configuration.md)
- [Architecture and Design](docs/architecture.md)

## Repository Layout

```text
chronos-finetuning/
  config/                        YAML configs for download and fine-tuning
  data/                          Example time-series datasets
  src/tachyon_model_downloader/  CLI modules and Pydantic schemas
  README.md
  LICENSE
  pyproject.toml
```

## Current Base Model

- Default `repo_id`: `amazon/chronos-bolt-base`

## License

This repository is licensed under the terms in `LICENSE`.
# tachyon-model

YAML-driven project to download Hugging Face base models locally with
idempotent behavior and state tracking.

## Default model

- `amazon/chronos-bolt-base`

## Project layout

- `config/model_download.yaml` - download configuration
- `src/tachyon_model_downloader/schemas.py` - Pydantic config schemas
- `src/tachyon_model_downloader/download_model.py` - CLI downloader
- `src/tachyon_model_downloader/fine_tune_and_export.py` - time-series tuning CLI

## Setup (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Setup (Bash)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Download command

```bash
python -m tachyon_model_downloader.download_model --config config/model_download.yaml
```

## Time-series fine-tuning sample

This project includes a sample `autogluon.timeseries` training flow using:

- `data/train.csv`
- `data/val.csv`
- `config/fine_tune.yaml`

Run:

```bash
poetry install
poetry run python -m tachyon_model_downloader.fine_tune_and_export
```

Outputs:

- Trained predictor: `artifacts/ag_predictor`
- Exported safetensors directory: `artifacts/final_safetensors`
- Training metadata: `artifacts/ag_predictor/training_state.json`

Chronos-specific tuning options are configured in `config/fine_tune.yaml`:

- `training.chronos_model_path` points to the local base model directory.
- `training.context_length`, `training.batch_size`, `training.max_epochs`, and
  `training.device` are passed to
  `hyperparameters["Chronos"]`.
- `training.fine_tune`, `training.fine_tune_lr`, `training.fine_tune_steps`, and
  `training.fine_tune_batch_size` control Chronos fine-tuning.

## Output

By default, the model downloads to:

- `artifacts/base-models/chronos-bolt-base`

The script writes a state file after successful completion:

- `artifacts/base-models/chronos-bolt-base/download_state.json`
# Chronos Bolt Offline Fine-Tuner

Download the Chronos Bolt forecasting model, fine-tune it on your own time-series data, and save a fully self-contained offline model for local inference.

## Features

- Download Chronos Bolt base model
- Fine-tune on custom datasets
- Export a complete offline model package
- Run local forecasts with CPU or GPU
- Reproducible training workflows
- Simple CLI tools
- Secure private deployment for air-gapped environments


## Use Cases

- Demand forecasting
- Sales prediction
- Inventory planning
- Sensor monitoring
- Financial trend analysis
- Research experiments

## Project Structure

```bash
chronos-finetuning/
├── data/
├── models/
├── outputs/
├── scripts/
│   ├── download_model.py
│   ├── train.py
│   ├── export_model.py
│   └── predict.py
├── configs/
├── requirements.txt
└── README.md
