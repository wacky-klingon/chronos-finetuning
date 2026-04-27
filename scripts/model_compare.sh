#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: scripts/model_compare.sh [--config <path>] [--run-id <id>] [--resume] [--force]"
  exit 1
}

CONFIG_PATH="config/model_compare.yaml"
RUN_ID=""
RESUME="false"
FORCE="false"
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
  "${PYTHON_CMD[@]}" -c "from pathlib import Path; from tachyon_model_downloader.model_compare_schemas import load_model_compare_config, resolve_path; cfg=load_model_compare_config(Path('$CONFIG_ABS')); root=Path('$PROJECT_ROOT'); print(resolve_path(root, cfg.output.output_root)); print(cfg.output.run_label); print(resolve_path(root, cfg.models.base_predictor_dir)); print(resolve_path(root, cfg.models.finetuned_predictor_dir)); print(resolve_path(root, cfg.query_data.history_csv)); print(resolve_path(root, cfg.query_data.targets_csv))"
)

if [[ "${#CONFIG_VALUES[@]}" -lt 6 ]]; then
  echo "Failed to parse config values from: $CONFIG_ABS"
  exit 8
fi

OUTPUT_ROOT="${CONFIG_VALUES[0]}"
RUN_LABEL="${CONFIG_VALUES[1]}"
BASE_PREDICTOR_DIR="${CONFIG_VALUES[2]}"
FINETUNED_PREDICTOR_DIR="${CONFIG_VALUES[3]}"
HISTORY_CSV="${CONFIG_VALUES[4]}"
TARGETS_CSV="${CONFIG_VALUES[5]}"

if [[ ! -d "$BASE_PREDICTOR_DIR" ]]; then
  echo "Config validation failed: base model directory does not exist."
  echo "Expected path: $BASE_PREDICTOR_DIR"
  echo "Update models.base_predictor_dir in: $CONFIG_ABS"
  exit 10
fi
if [[ ! -f "${BASE_PREDICTOR_DIR}/config.json" || ! -f "${BASE_PREDICTOR_DIR}/model.safetensors" ]]; then
  echo "Config validation failed: base model directory is missing offline artifacts."
  echo "Required files: ${BASE_PREDICTOR_DIR}/config.json and ${BASE_PREDICTOR_DIR}/model.safetensors"
  echo "Update models.base_predictor_dir in: $CONFIG_ABS"
  exit 10
fi
if [[ ! -d "$FINETUNED_PREDICTOR_DIR" ]]; then
  echo "Config validation failed: fine-tuned model directory does not exist."
  echo "Expected path: $FINETUNED_PREDICTOR_DIR"
  echo "Update models.finetuned_predictor_dir in: $CONFIG_ABS"
  exit 11
fi
if [[ ! -f "${FINETUNED_PREDICTOR_DIR}/config.json" || ! -f "${FINETUNED_PREDICTOR_DIR}/model.safetensors" ]]; then
  echo "Config validation failed: fine-tuned model directory is missing offline artifacts."
  echo "Required files: ${FINETUNED_PREDICTOR_DIR}/config.json and ${FINETUNED_PREDICTOR_DIR}/model.safetensors"
  echo "Update models.finetuned_predictor_dir in: $CONFIG_ABS"
  exit 11
fi
if [[ ! -f "$HISTORY_CSV" ]]; then
  echo "Config validation failed: history CSV does not exist."
  echo "Expected path: $HISTORY_CSV"
  echo "Update query_data.history_csv in: $CONFIG_ABS"
  exit 12
fi
if [[ ! -f "$TARGETS_CSV" ]]; then
  echo "Config validation failed: targets CSV does not exist."
  echo "Expected path: $TARGETS_CSV"
  echo "Update query_data.targets_csv in: $CONFIG_ABS"
  exit 13
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$("${PYTHON_CMD[@]}" -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))")_${RUN_LABEL}"
fi

RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
BASE_OUTPUT="${RUN_DIR}/base_predictions.jsonl"
FINETUNED_OUTPUT="${RUN_DIR}/finetuned_predictions.jsonl"
COMPARISON_JSON="${RUN_DIR}/comparison.json"
COMPARISON_MD="${RUN_DIR}/comparison.md"
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

step_log "Completed run."
step_log "Base predictions: $BASE_OUTPUT"
step_log "Fine-tuned predictions: $FINETUNED_OUTPUT"
step_log "Comparison JSON: $COMPARISON_JSON"
step_log "Comparison markdown: $COMPARISON_MD"
