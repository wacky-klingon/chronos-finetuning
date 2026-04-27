# Configuration Reference

The pipeline is fully config-driven. Runtime behavior is controlled by:

- `config/model_download.yaml`
- `config/fine_tune.yaml`
- `config/model_compare.yaml`

All input data must be in parquet format. All model directories must be in
full offline safetensors mode (see "Safetensors-only model policy" below).

## `config/model_download.yaml`

Example:

```yaml
model:
  repo_id: "amazon/chronos-bolt-base"
  revision: null
  allow_patterns: null
  ignore_patterns: null

download:
  output_dir: "./artifacts/base-models/chronos-bolt-base"
  force_download: false
  local_dir_use_symlinks: false
```

Fields:

- `model.repo_id`: Hugging Face model repository ID.
- `model.revision`: Optional branch, tag, or commit.
- `model.allow_patterns`: Optional include filter for snapshot files.
- `model.ignore_patterns`: Optional exclude filter for snapshot files.
- `download.output_dir`: Relative or absolute target directory.
- `download.force_download`: If `true`, re-download even if files exist.
- `download.local_dir_use_symlinks`: Controls local snapshot symlink behavior.

## `config/fine_tune.yaml`

Example:

```yaml
data:
  train_parquet: "./data/train.parquet"
  val_parquet: "./data/val.parquet"

schema:
  use_parquet_metadata: true
  json_schema_path: null
  role_hints:
    item_id_column: "item_id"
    timestamp_column: "timestamp"
    target_column: "target"
    known_covariate_columns: []
    past_covariate_columns: []
    static_covariate_columns: []

training:
  prediction_length: 12
  eval_metric: "MASE"
  presets: "medium_quality"
  chronos_model_path: "./artifacts/base-models/chronos-bolt-base"
  context_length: 64
  batch_size: 16
  max_epochs: 3
  device: "cpu"
  fine_tune: true
  fine_tune_lr: 0.00001
  fine_tune_steps: 200
  fine_tune_batch_size: 8

paths:
  predictor_dir: "./artifacts/ag_predictor"
  export_dir: "./artifacts/final_safetensors"
```

Fields:

- `data.train_parquet`: Training parquet file path (must end with `.parquet`).
- `data.val_parquet`: Validation parquet file path (must end with `.parquet`).
- `schema.*`: Dataset role resolution. See "Schema role resolution" below.
- `training.prediction_length`: Forecast horizon.
- `training.eval_metric`: AutoGluon evaluation metric.
- `training.presets`: AutoGluon training preset profile.
- `training.chronos_model_path`: Downloaded base model directory.
- `training.context_length`: Context window used by Chronos model training.
- `training.batch_size`: Training batch size.
- `training.max_epochs`: Maximum number of training epochs.
- `training.device`: Device target such as `cpu` or `cuda`.
- `training.fine_tune`: Enables or disables fine-tuning behavior.
- `training.fine_tune_lr`: Fine-tuning learning rate.
- `training.fine_tune_steps`: Fine-tuning step budget.
- `training.fine_tune_batch_size`: Batch size for fine-tuning loop.
- `paths.predictor_dir`: AutoGluon predictor output directory.
- `paths.export_dir`: Offline `safetensors` export directory.

## Path resolution rules

- Relative paths are resolved from the repository root.
- Absolute paths are accepted and used directly.
- Keep paths in YAML rather than hardcoding values in Python code.

## `config/model_compare.yaml`

Example:

```yaml
query_data:
  history_parquet: "./data/history.parquet"
  targets_parquet: "./data/targets.parquet"

schema:
  use_parquet_metadata: true
  json_schema_path: null
  role_hints:
    item_id_column: "item_id"
    timestamp_column: "timestamp"
    target_column: "target"
    known_covariate_columns: []
    past_covariate_columns: []
    static_covariate_columns: []

sampling:
  temperature: 1.0
  top_p: 0.9
  num_samples: 100
  seed: 42
  prediction_length: 12
  lower_quantile: 0.1
  median_quantile: 0.5
  upper_quantile: 0.9

models:
  base_model_dir: "./artifacts/base-models/chronos-bolt-base"
  finetuned_model_dir: "./artifacts/final_safetensors"
  device: "cpu"

output:
  output_root: "./outputs/model_compare"
  run_label: "chronos_bolt"

analysis_hooks:
  enabled: false
  provider: "none"
  codex:
    api_base: "https://api.openai.com/v1"
    model: "gpt-4o-mini"
    api_key_env: "OPENAI_API_KEY"
    max_output_tokens: 2000
    request_timeout_seconds: 60
```

Fields:

- `query_data.history_parquet`: History parquet file path.
- `query_data.targets_parquet`: Targets (held-out) parquet file path.
- `schema.*`: Dataset role resolution. See "Schema role resolution" below.
- `sampling.temperature`: Recorded sampling temperature metadata for the run.
- `sampling.top_p`: Recorded nucleus sampling metadata for the run.
- `sampling.num_samples`: Recorded sampling count metadata for the run.
- `sampling.seed`: Sampling seed metadata for reproducibility.
- `sampling.prediction_length`: Forecast horizon used for inference.
- `sampling.lower_quantile`: Numeric lower-bound quantile (e.g., `0.1`).
- `sampling.median_quantile`: Numeric median quantile (e.g., `0.5`).
- `sampling.upper_quantile`: Numeric upper-bound quantile (e.g., `0.9`).
- `models.base_model_dir`: Offline safetensors model directory for the baseline.
- `models.finetuned_model_dir`: Offline safetensors model directory for the fine-tuned model.
- `models.device`: Inference device, `cpu` or `cuda`.
- `output.output_root`: Root directory for generated comparison runs.
- `output.run_label`: Label appended to auto-generated run id.
- `analysis_hooks.enabled`: When `true`, attempt downstream analysis after compare.
- `analysis_hooks.provider`: Analysis provider, `codex` or `none`.
- `analysis_hooks.codex.*`: Codex API configuration (see "Analysis hooks").

## Schema role resolution

Dataset role mappings are resolved in this priority order:

1. **Embedded parquet metadata.** When `schema.use_parquet_metadata` is `true`,
   the loader looks for the schema metadata key `chronos_roles` whose value is
   a UTF-8 JSON object describing the roles. Producers can embed this when
   writing parquet via pyarrow.
2. **JSON schema fallback.** If `schema.json_schema_path` is set (relative or
   absolute), the loader reads a JSON file of the form:

   ```json
   {
     "roles": {
       "item_id_column": "item_id",
       "timestamp_column": "timestamp",
       "target_column": "target",
       "known_covariate_columns": [],
       "past_covariate_columns": [],
       "static_covariate_columns": []
     }
   }
   ```
3. **Inline `role_hints`.** If neither metadata nor a JSON schema file is
   available, the inline `schema.role_hints` block in YAML is used.

Resolution is deterministic and idempotent: identical inputs and config always
yield identical role assignments. If none of the sources provide a valid role
mapping, the run fails fast with an actionable error.

### Covariate roles

- `known_covariate_columns`: Variables known into the future (for example
  promotional flags). May be empty for v1 baseline runs.
- `past_covariate_columns`: Variables observed historically only.
- `static_covariate_columns`: Time-invariant attributes.

## Safetensors-only model policy

The pipeline operates in **full offline mode** and only loads safetensors
artifacts.

A valid model directory must:

- contain `config.json`
- contain `model.safetensors`
- not contain any `.pkl`, `.pickle`, or `.bin` files anywhere in the directory
  tree

The fine-tune export step automatically purges these forbidden artifacts after
running `save_pretrained`. The compare step rejects any model directory that
contains them. AutoGluon predictor pickle files are not loaded at inference
time.

## Analysis hooks

When `analysis_hooks.enabled` is `false` or `provider` is `none`, the compare
flow still emits prompt-export artifacts:

- `<run>/analysis/analysis_payload.json` - bundled raw metric and run context
- `<run>/analysis/analysis_prompt.md` - copy-ready prompt for an external LLM
- `<run>/analysis/analysis_hook_status.json` - hook execution status

When `analysis_hooks.enabled` is `true` and `provider` is `codex`, the runner
additionally calls the Codex API directly using the chat-completions endpoint
defined by `analysis_hooks.codex.api_base`. The API key is read from the
environment variable named in `analysis_hooks.codex.api_key_env` (default
`OPENAI_API_KEY`). Credentials must never be committed to YAML.

The provider response, when produced, is written to
`<run>/analysis/analysis_response.md`.

The pipeline itself never produces win/lose, pass/fail, or recommendation
labels. Interpretation is delegated entirely to the analysis hook consumer.
