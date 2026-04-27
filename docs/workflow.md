# Pipeline Workflow

This project produces baseline-vs-finetuned probabilistic forecast metrics for
Chronos models. The workflow is parquet-first, schema-driven, and operates on
safetensors-only model directories.

## Stage 1: Base model acquisition

Command:

```bash
poetry run python -m tachyon_model_downloader.download_model --config config/model_download.yaml
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
poetry run python -m tachyon_model_downloader.fine_tune_and_export --config config/fine_tune.yaml
```

What happens:

1. Fine-tuning config is loaded and validated.
2. Train and validation parquet files are loaded.
3. Dataset roles are resolved using the configured schema source priority
   (parquet metadata first, then JSON schema, then inline `role_hints`).
4. Required columns are checked against the resolved roles.
5. Data is converted into `TimeSeriesDataFrame`.
6. `TimeSeriesPredictor.fit(...)` runs with Chronos hyperparameters.
7. Predictor artifacts are persisted in `paths.predictor_dir`.
8. A `save_pretrained`-compatible model object is discovered and exported as
   safetensors plus tokenizer/processor artifacts in `paths.export_dir`.
9. Pickle-based artifacts (`.pkl`, `.pickle`, `.bin`) emitted by the exporter
   are purged from the export directory.
10. The export directory is validated against the safetensors-only policy.
11. State and metrics JSON reports are written, including a
    `safetensors_compliant` flag.

## Stage 3: Compare base vs fine-tuned

Command:

```bash
poetry run bash scripts/model_compare.sh --config config/model_compare.yaml
```

Useful flags:

- `--run-id <id>`: override the auto-generated run id.
- `--resume`: reuse existing run artifacts when present.
- `--force`: delete the existing run directory before starting.
- `--skip-hooks`: skip the analysis hook step entirely.
- `--skip-direct-call`: run hooks but only emit prompt-export artifacts.
- `--schema-path <path>`: convenience override for the schema JSON path.

What happens:

1. Pre-run validation:
   - Both `models.base_model_dir` and `models.finetuned_model_dir` exist.
   - Each model directory contains `config.json` and `model.safetensors`.
   - Neither model directory contains `.pkl`, `.pickle`, or `.bin` artifacts.
   - Both `query_data.history_parquet` and `query_data.targets_parquet` are
     parquet files.
2. Inference is run for the base model role, writing
   `<run>/base_predictions.jsonl`.
3. Inference is run for the fine-tuned model role, writing
   `<run>/finetuned_predictions.jsonl`.
4. Inference loads each model directly via `BaseChronosPipeline.from_pretrained`
   from its safetensors directory; it does not load AutoGluon predictor
   pickles.
5. Per-series metrics are emitted: average logprob, average abs error, average
   interval width, interval coverage, pinball losses (lower/median/upper),
   latency seconds.
6. Comparison generation produces:
   - `<run>/comparison.json` - structured comparison summary
   - `<run>/comparison.md` - human-readable raw metric table
7. Analysis hooks step (unless `--skip-hooks`):
   - `<run>/analysis/analysis_payload.json`
   - `<run>/analysis/analysis_prompt.md`
   - `<run>/analysis/analysis_hook_status.json`
   - When `analysis_hooks.enabled` and credentials are present, also
     `<run>/analysis/analysis_response.md`.

## Idempotency behavior

- Base model download is idempotent by default because the downloader checks
  for `config.json` in the target model directory before downloading.
- Fine-tuning writes outputs to deterministic directories, so repeated runs
  update existing output locations instead of creating random paths.
- Compare runs use deterministic run-ids and support `--resume`.
- Schema role resolution is deterministic: identical inputs always yield
  identical role mappings.

## Data contract

Input parquet files must include the columns identified by the resolved roles.
Required roles are:

- `item_id_column` (time-series entity key)
- `timestamp_column` (parseable datetime)
- `target_column` (numeric value for forecasting)

Optional covariate roles are supported via `known_covariate_columns`,
`past_covariate_columns`, and `static_covariate_columns`. See
[`configuration.md`](configuration.md) for full details.
