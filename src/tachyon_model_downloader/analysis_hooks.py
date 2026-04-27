"""Analysis hook router for downstream interpretation of raw metric artifacts.

Always writes a prompt-export fallback bundle (``analysis_payload.json`` and
``analysis_prompt.md``) so any IDE/Codex/ChatGPT consumer can ingest the same
payload manually. When ``analysis_hooks.enabled`` is true and the configured
provider credentials are available, an optional direct API call is dispatched
and its response is written to ``analysis_response.md``.

This module deliberately performs NO interpretive comparison logic; it only
packages the raw metric outputs for downstream consumers.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .dataset_schema import DatasetRoles, resolve_dataset_roles
from .model_compare_schemas import (
    AnalysisHooksConfig,
    CodexHookConfig,
    ModelCompareConfig,
    load_model_compare_config,
    resolve_path,
)

PROMPT_EXPORT_PAYLOAD_FILENAME = "analysis_payload.json"
PROMPT_EXPORT_PROMPT_FILENAME = "analysis_prompt.md"
PROVIDER_RESPONSE_FILENAME = "analysis_response.md"
HOOK_STATUS_FILENAME = "analysis_hook_status.json"


class AnalysisPayload(BaseModel):
    generated_at_utc: str
    run_label: str
    base_model_dir: str
    finetuned_model_dir: str
    history_parquet: str
    targets_parquet: str
    resolved_roles: DatasetRoles
    sampling: dict[str, Any]
    comparison: dict[str, Any]


class AnalysisHookStatus(BaseModel):
    generated_at_utc: str
    enabled: bool
    provider: Literal["codex", "none"]
    payload_path: str
    prompt_path: str
    response_path: str | None
    direct_call_attempted: bool
    direct_call_succeeded: bool
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build analysis payload and optionally call the configured provider."
    )
    parser.add_argument("--config", required=True, help="Path to model_compare YAML config.")
    parser.add_argument(
        "--comparison-json", required=True, help="Path to comparison.json artifact."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where analysis artifacts will be written.",
    )
    parser.add_argument(
        "--skip-direct-call",
        action="store_true",
        help="Skip provider API call and emit fallback artifacts only.",
    )
    return parser.parse_args()


def _read_comparison_json(comparison_json_path: Path) -> dict[str, Any]:
    if not comparison_json_path.exists():
        raise FileNotFoundError(f"Comparison JSON not found: {comparison_json_path}")
    with comparison_json_path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    if not isinstance(data, dict):
        raise ValueError(f"Comparison JSON must be a JSON object: {comparison_json_path}")
    return data


def build_payload(
    config: ModelCompareConfig,
    project_root: Path,
    comparison_json_path: Path,
) -> AnalysisPayload:
    history_path = resolve_path(project_root, config.query_data.history_parquet)
    targets_path = resolve_path(project_root, config.query_data.targets_parquet)
    base_model_dir = resolve_path(project_root, config.models.base_model_dir)
    finetuned_model_dir = resolve_path(project_root, config.models.finetuned_model_dir)
    roles = resolve_dataset_roles(
        parquet_path=history_path,
        schema_source=config.schema_source,
        project_root=project_root,
    )
    comparison = _read_comparison_json(comparison_json_path=comparison_json_path)
    return AnalysisPayload(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        run_label=config.output.run_label,
        base_model_dir=str(base_model_dir),
        finetuned_model_dir=str(finetuned_model_dir),
        history_parquet=str(history_path),
        targets_parquet=str(targets_path),
        resolved_roles=roles,
        sampling=config.sampling.model_dump(),
        comparison=comparison,
    )


def render_prompt_markdown(payload: AnalysisPayload) -> str:
    lines: list[str] = [
        "# Chronos Base vs Fine-Tuned Analysis Request",
        "",
        "You are given raw metric outputs from a base vs fine-tuned Chronos comparison.",
        "Provide an interpretation that highlights:",
        "",
        "- whether the fine-tuned model improved or regressed across reported metrics,",
        "- where uncertainty calibration changed (interval coverage and width),",
        "- any per-series outliers worth investigating,",
        "- caveats based on the configured sampling and prediction length.",
        "",
        "Do not fabricate metrics. Use only the JSON payload appended below.",
        "",
        "## Run context",
        "",
        f"- Run label: {payload.run_label}",
        f"- Generated at UTC: {payload.generated_at_utc}",
        f"- Base model dir: {payload.base_model_dir}",
        f"- Fine-tuned model dir: {payload.finetuned_model_dir}",
        f"- History parquet: {payload.history_parquet}",
        f"- Targets parquet: {payload.targets_parquet}",
        "",
        "## Resolved dataset roles",
        "",
        "```json",
        json.dumps(payload.resolved_roles.model_dump(), indent=2),
        "```",
        "",
        "## Sampling configuration",
        "",
        "```json",
        json.dumps(payload.sampling, indent=2),
        "```",
        "",
        "## Comparison artifact",
        "",
        "```json",
        json.dumps(payload.comparison, indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


def write_prompt_export(payload: AnalysisPayload, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / PROMPT_EXPORT_PAYLOAD_FILENAME
    prompt_path = output_dir / PROMPT_EXPORT_PROMPT_FILENAME
    with payload_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload.model_dump(), file_handle, indent=2)
    with prompt_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write(render_prompt_markdown(payload=payload))
    return payload_path, prompt_path


def _read_api_key(env_var_name: str) -> str | None:
    raw = os.environ.get(env_var_name)
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    return stripped


def _build_chat_request_body(payload: AnalysisPayload, codex_config: CodexHookConfig) -> bytes:
    user_prompt = render_prompt_markdown(payload=payload)
    body: dict[str, Any] = {
        "model": codex_config.model,
        "max_tokens": codex_config.max_output_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You analyze raw probabilistic forecasting metrics and provide "
                    "concise, evidence-grounded interpretations. You never invent "
                    "metric values that are not present in the supplied JSON."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    }
    return json.dumps(body).encode("utf-8")


def _post_chat_completion(
    codex_config: CodexHookConfig, api_key: str, body: bytes
) -> dict[str, Any]:
    parsed_api_base = urllib.parse.urlparse(codex_config.api_base)
    if parsed_api_base.scheme != "https":
        raise ValueError("analysis_hooks.codex.api_base must use https:// for direct calls.")
    if not parsed_api_base.netloc:
        raise ValueError(
            "analysis_hooks.codex.api_base must include a valid host for direct calls."
        )
    url = codex_config.api_base.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(  # nosec B310 - scheme is validated as https above
        request, timeout=codex_config.request_timeout_seconds
    ) as response:
        raw = response.read().decode("utf-8")
    parsed: Any = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Codex API response must be a JSON object.")
    return parsed


def _extract_assistant_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return json.dumps(response_payload, indent=2)
    first = choices[0]
    if not isinstance(first, dict):
        return json.dumps(response_payload, indent=2)
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return json.dumps(response_payload, indent=2)


def call_codex(
    payload: AnalysisPayload,
    codex_config: CodexHookConfig,
    output_dir: Path,
) -> tuple[bool, Path | None, str]:
    api_key = _read_api_key(codex_config.api_key_env)
    if api_key is None:
        return (
            False,
            None,
            f"Skipping direct Codex call: env var {codex_config.api_key_env} not set.",
        )
    request_body = _build_chat_request_body(payload=payload, codex_config=codex_config)
    try:
        response_payload = _post_chat_completion(
            codex_config=codex_config, api_key=api_key, body=request_body
        )
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        ValueError,
    ) as exc:
        return (False, None, f"Codex API call failed: {exc}")
    except json.JSONDecodeError as exc:
        return (False, None, f"Codex API returned non-JSON response: {exc}")
    response_text = _extract_assistant_text(response_payload=response_payload)
    response_path = output_dir / PROVIDER_RESPONSE_FILENAME
    with response_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write(response_text)
    return (True, response_path, "Codex API call succeeded.")


def write_status(status: AnalysisHookStatus, output_dir: Path) -> Path:
    status_path = output_dir / HOOK_STATUS_FILENAME
    with status_path.open("w", encoding="utf-8") as file_handle:
        json.dump(status.model_dump(), file_handle, indent=2)
    return status_path


def run_analysis_hooks(
    config: ModelCompareConfig,
    project_root: Path,
    comparison_json_path: Path,
    output_dir: Path,
    skip_direct_call: bool,
) -> AnalysisHookStatus:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        config=config,
        project_root=project_root,
        comparison_json_path=comparison_json_path,
    )
    payload_path, prompt_path = write_prompt_export(payload=payload, output_dir=output_dir)
    hooks_config: AnalysisHooksConfig = config.analysis_hooks
    direct_call_attempted = False
    direct_call_succeeded = False
    response_path: Path | None = None
    message = "Prompt-export fallback artifacts written."
    if (
        hooks_config.enabled
        and hooks_config.provider == "codex"
        and hooks_config.codex is not None
        and not skip_direct_call
    ):
        direct_call_attempted = True
        direct_call_succeeded, response_path, message = call_codex(
            payload=payload,
            codex_config=hooks_config.codex,
            output_dir=output_dir,
        )
    status = AnalysisHookStatus(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        enabled=hooks_config.enabled,
        provider=hooks_config.provider,
        payload_path=str(payload_path),
        prompt_path=str(prompt_path),
        response_path=str(response_path) if response_path is not None else None,
        direct_call_attempted=direct_call_attempted,
        direct_call_succeeded=direct_call_succeeded,
        message=message,
    )
    write_status(status=status, output_dir=output_dir)
    return status


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    config = load_model_compare_config(config_path=config_path)
    comparison_json_path = Path(args.comparison_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    status = run_analysis_hooks(
        config=config,
        project_root=project_root,
        comparison_json_path=comparison_json_path,
        output_dir=output_dir,
        skip_direct_call=bool(args.skip_direct_call),
    )
    print(f"Analysis payload: {status.payload_path}")
    print(f"Analysis prompt:  {status.prompt_path}")
    if status.response_path is not None:
        print(f"Analysis response: {status.response_path}")
    print(f"Status: {status.message}")


if __name__ == "__main__":
    main()
