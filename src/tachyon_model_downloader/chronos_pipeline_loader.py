"""Load Chronos pipelines directly from safetensors offline directories.

Loading is performed via ``chronos.BaseChronosPipeline.from_pretrained`` so that
no AutoGluon predictor pickle files are required at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from chronos import BaseChronosPipeline

from .model_validation import validate_safetensors_only_model_dir

SUPPORTED_DEVICES: tuple[str, ...] = ("cpu", "cuda")


def load_chronos_pipeline(model_dir: Path, device: str, role_label: str) -> Any:
    """Validate and load a Chronos pipeline from a safetensors-only directory."""
    validate_safetensors_only_model_dir(model_dir=model_dir, role_label=role_label)
    if device not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported device '{device}' for {role_label}. "
            f"Supported devices: {SUPPORTED_DEVICES}"
        )
    return BaseChronosPipeline.from_pretrained(
        str(model_dir),
        device_map=device,
        dtype=torch.float32,
    )
