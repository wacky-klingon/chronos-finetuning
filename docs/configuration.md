# Configuration Reference

Chronos Fine-Tuning is config-driven. Runtime behavior is controlled by:

- `config/model_download.yaml`
- `config/fine_tune.yaml`

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
  train_csv: "./data/train.csv"
  val_csv: "./data/val.csv"
  item_id_column: "item_id"
  timestamp_column: "timestamp"
  target_column: "target"

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

- `data.*`: Input data locations and required column names.
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
