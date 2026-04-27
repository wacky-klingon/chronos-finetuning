#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: scripts/model_compare.sh [options]"
  echo "  --config <path>        Path to model_compare.yaml (default: config/model_compare.yaml)"
  echo "  --run-id <id>          Override auto-generated run id"
  echo "  --resume               Reuse existing run artifacts when available"
  echo "  --force                Delete existing run directory before starting"
  echo "  --skip-hooks           Skip analysis hook step entirely"
  echo "  --skip-direct-call     Run hooks but only emit prompt-export artifacts"
  echo "  --schema-path <path>   Override schema.json_schema_path for the run"
  exit 1
}

CONFIG_PATH="config/model_compare.yaml"
RUN_ID=""
RESUME="false"
FORCE="false"
SKIP_HOOKS="false"
SKIP_DIRECT_CALL="false"
SCHEMA_PATH_OVERRIDE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_CMD=(poetry run python)

if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry is required but was not found on PATH."
  echo "Install Poetry, then run: poetry install"
  exit 9
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --resume)
      RESUME="true"
      shift
      ;;
    --force)
      FORCE="true"
      shift
      ;;
    --skip-hooks)
      SKIP_HOOKS="true"
      shift
      ;;
    --skip-direct-call)
      SKIP_DIRECT_CALL="true"
      shift
      ;;
    --schema-path)
      SCHEMA_PATH_OVERRIDE="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

if [[ "$RESUME" == "true" && "$FORCE" == "true" ]]; then
  echo "Cannot use --resume and --force together."
  exit 2
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH"
  exit 3
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "$SCHEMA_PATH_OVERRIDE" ]]; then
  export TACHYON_SCHEMA_PATH_OVERRIDE="$SCHEMA_PATH_OVERRIDE"
fi

CONFIG_ABS="$("${PYTHON_CMD[@]}" -c "from pathlib import Path; print(Path('$CONFIG_PATH').resolve())")"
PROJECT_ROOT="$("${PYTHON_CMD[@]}" -c "from pathlib import Path; p=Path('$CONFIG_ABS').resolve(); print(p.parent.parent)")"

if ! "${PYTHON_CMD[@]}" -c "import tachyon_model_downloader" >/dev/null 2>&1; then
  echo "Unable to import 'tachyon_model_downloader'."
  echo "Ensure dependencies are installed and src is available on PYTHONPATH."
  echo "Suggested setup:"
  echo "  poetry install"
  exit 7
fi

readarray -t CONFIG_VALUES < <(
  "${PYTHON_CMD[@]}" -c "from pathlib import Path; from tachyon_model_downloader.model_compare_schemas import load_model_compare_config, resolve_path; cfg=load_model_compare_config(Path('$CONFIG_ABS')); root=Path('$PROJECT_ROOT'); print(resolve_path(root, cfg.output.output_root)); print(cfg.output.run_label); print(resolve_path(root, cfg.models.base_model_dir)); print(resolve_path(root, cfg.models.finetuned_model_dir)); print(resolve_path(root, cfg.query_data.history_parquet)); print(resolve_path(root, cfg.query_data.targets_parquet)); print('1' if cfg.analysis_hooks.enabled else '0')"
)

if [[ "${#CONFIG_VALUES[@]}" -lt 7 ]]; then
  echo "Failed to parse config values from: $CONFIG_ABS"
  exit 8
fi

OUTPUT_ROOT="${CONFIG_VALUES[0]}"
RUN_LABEL="${CONFIG_VALUES[1]}"
BASE_MODEL_DIR="${CONFIG_VALUES[2]}"
FINETUNED_MODEL_DIR="${CONFIG_VALUES[3]}"
HISTORY_PARQUET="${CONFIG_VALUES[4]}"
TARGETS_PARQUET="${CONFIG_VALUES[5]}"
HOOKS_ENABLED="${CONFIG_VALUES[6]}"

validate_safetensors_only() {
  local label="$1"
  local dir="$2"
  if [[ ! -d "$dir" ]]; then
    echo "Config validation failed: ${label} model directory does not exist."
    echo "Expected path: $dir"
    exit 10
  fi
  if [[ ! -f "${dir}/config.json" || ! -f "${dir}/model.safetensors" ]]; then
    echo "Config validation failed: ${label} model directory missing required safetensors artifacts."
    echo "Required files: ${dir}/config.json and ${dir}/model.safetensors"
    exit 10
  fi
  if find "$dir" -type f \( -iname '*.pkl' -o -iname '*.pickle' -o -iname '*.bin' \) | read -r _; then
    echo "Config validation failed: ${label} model directory contains pickle-based artifacts."
    echo "The safetensors-only offline mode does not allow .pkl, .pickle, or .bin files."
    echo "Offending directory: $dir"
    exit 10
  fi
}

validate_parquet() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "Config validation failed: ${label} parquet file does not exist."
    echo "Expected path: $path"
    exit 12
  fi
  case "${path,,}" in
    *.parquet) ;;
    *)
      echo "Config validation failed: ${label} input must have a .parquet extension."
      echo "Got: $path"
      exit 12
      ;;
  esac
}

validate_safetensors_only "base" "$BASE_MODEL_DIR"
validate_safetensors_only "fine-tuned" "$FINETUNED_MODEL_DIR"
validate_parquet "history" "$HISTORY_PARQUET"
validate_parquet "targets" "$TARGETS_PARQUET"

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$("${PYTHON_CMD[@]}" -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))")_${RUN_LABEL}"
fi

RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
BASE_OUTPUT="${RUN_DIR}/base_predictions.jsonl"
FINETUNED_OUTPUT="${RUN_DIR}/finetuned_predictions.jsonl"
COMPARISON_JSON="${RUN_DIR}/comparison.json"
COMPARISON_MD="${RUN_DIR}/comparison.md"
HOOKS_DIR="${RUN_DIR}/analysis"
LOCK_FILE="${RUN_DIR}/.lock"

if [[ "$FORCE" == "true" && -d "$RUN_DIR" ]]; then
  rm -rf "$RUN_DIR"
fi
mkdir -p "$LOG_DIR"

if [[ -f "$LOCK_FILE" ]]; then
  echo "Run lock exists: $LOCK_FILE"
  exit 4
fi
trap 'rm -f "$LOCK_FILE"' EXIT
touch "$LOCK_FILE"

step_log() {
  local message="$1"
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $message"
}

run_step() {
  local step_name="$1"
  local log_file="$2"
  shift 2
  step_log "Running ${step_name}."
  if ! "$@" >"$log_file" 2>&1; then
    echo "Step failed: ${step_name}"
    echo "Log file: $log_file"
    echo "Last 40 log lines:"
    if [[ -f "$log_file" ]]; then
      tail -n 40 "$log_file"
    else
      echo "Log file not found."
    fi
    exit 14
  fi
}

is_valid_jsonl() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  return 0
}

step_log "Starting model comparison run: $RUN_ID"
step_log "Run directory: $RUN_DIR"

if [[ "$RESUME" == "true" && "$(is_valid_jsonl "$BASE_OUTPUT"; echo $?)" -eq 0 ]]; then
  step_log "Resume mode: skipping base inference."
else
  run_step "base inference" "${LOG_DIR}/base_inference.log" "${PYTHON_CMD[@]}" -m tachyon_model_downloader.run_inference \
    --config "$CONFIG_ABS" \
    --model-role base \
    --output-file "$BASE_OUTPUT"
fi

if [[ "$RESUME" == "true" && "$(is_valid_jsonl "$FINETUNED_OUTPUT"; echo $?)" -eq 0 ]]; then
  step_log "Resume mode: skipping fine-tuned inference."
else
  run_step "fine-tuned inference" "${LOG_DIR}/finetuned_inference.log" "${PYTHON_CMD[@]}" -m tachyon_model_downloader.run_inference \
    --config "$CONFIG_ABS" \
    --model-role finetuned \
    --output-file "$FINETUNED_OUTPUT"
fi

BASE_LINES="$("${PYTHON_CMD[@]}" -c "from pathlib import Path; p=Path('$BASE_OUTPUT'); print(sum(1 for line in p.open('r', encoding='utf-8') if line.strip()) if p.exists() else 0)")"
FINETUNED_LINES="$("${PYTHON_CMD[@]}" -c "from pathlib import Path; p=Path('$FINETUNED_OUTPUT'); print(sum(1 for line in p.open('r', encoding='utf-8') if line.strip()) if p.exists() else 0)")"

if [[ "$BASE_LINES" -eq 0 || "$FINETUNED_LINES" -eq 0 ]]; then
  echo "Inference output missing or empty."
  exit 5
fi
if [[ "$BASE_LINES" -ne "$FINETUNED_LINES" ]]; then
  echo "Inference output line counts differ: base=$BASE_LINES finetuned=$FINETUNED_LINES"
  exit 6
fi

if [[ "$RESUME" == "true" && -s "$COMPARISON_JSON" && -s "$COMPARISON_MD" ]]; then
  step_log "Resume mode: skipping comparison generation."
else
  run_step "comparison generation" "${LOG_DIR}/comparison.log" "${PYTHON_CMD[@]}" -m tachyon_model_downloader.compare_outputs \
    --base-file "$BASE_OUTPUT" \
    --finetuned-file "$FINETUNED_OUTPUT" \
    --output-json "$COMPARISON_JSON" \
    --output-markdown "$COMPARISON_MD"
fi

if [[ "$SKIP_HOOKS" == "true" ]]; then
  step_log "Skipping analysis hook step (--skip-hooks)."
else
  HOOK_ARGS=("${PYTHON_CMD[@]}" -m tachyon_model_downloader.analysis_hooks
    --config "$CONFIG_ABS"
    --comparison-json "$COMPARISON_JSON"
    --output-dir "$HOOKS_DIR")
  if [[ "$SKIP_DIRECT_CALL" == "true" ]]; then
    HOOK_ARGS+=(--skip-direct-call)
  fi
  if [[ "$RESUME" == "true" && -s "${HOOKS_DIR}/analysis_payload.json" && -s "${HOOKS_DIR}/analysis_prompt.md" ]]; then
    step_log "Resume mode: skipping analysis hook step."
  else
    run_step "analysis hooks" "${LOG_DIR}/analysis_hooks.log" "${HOOK_ARGS[@]}"
  fi
fi

step_log "Completed run."
step_log "Base predictions: $BASE_OUTPUT"
step_log "Fine-tuned predictions: $FINETUNED_OUTPUT"
step_log "Comparison JSON: $COMPARISON_JSON"
step_log "Comparison markdown: $COMPARISON_MD"
if [[ "$SKIP_HOOKS" != "true" ]]; then
  step_log "Analysis artifacts: $HOOKS_DIR"
fi
