# Pipeline Workflow

This project uses a two-stage pipeline for reproducible Chronos fine-tuning.

## Stage 1: Base model acquisition

Command:

```bash
python -m tachyon_model_downloader.download_model --config config/model_download.yaml
```

What happens:

1. YAML config is validated using Pydantic models.
2. Target output directory is created if missing.
3. Existing downloads are reused unless `force_download` is enabled.
4. Hugging Face snapshot files are downloaded.
5. `download_state.json` is written for run tracking.

## Stage 2: Fine-tune and export

Command:

```bash
python -m tachyon_model_downloader.fine_tune_and_export
```

What happens:

1. Fine-tuning config is loaded and validated.
2. Train and validation CSV files are loaded and schema-checked.
3. Data is converted into `TimeSeriesDataFrame`.
4. `TimeSeriesPredictor.fit(...)` runs with Chronos hyperparameters.
5. Predictor artifacts are persisted in `paths.predictor_dir`.
6. A save-pretrained-compatible model object is discovered.
7. Model and tokenizer/processor artifacts are exported to `paths.export_dir`.
8. State and metrics JSON reports are written.

## Idempotency behavior

- Base model download is idempotent by default because the downloader checks for
  `config.json` in the target model directory before downloading.
- Fine-tuning writes outputs to deterministic directories, so repeated runs update
  existing output locations instead of creating random paths.

## Data contract

Input train and validation CSV files must include:

- `item_id` column (time-series entity key)
- `timestamp` column (parseable datetime)
- `target` column (numeric value for forecasting)

Column names can be changed through `config/fine_tune.yaml`.
