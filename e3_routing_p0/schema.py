"""The stable P0 routing-record contract."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = "e3.routing_record.v1"
SUPPORTED_FAMILIES = frozenset({"moe", "mot", "latent"})
USAGE_SEMANTICS = {
    "moe": "topk_selection_share",
    "mot": "mean_mixture_probability",
    "latent": "mean_mixture_probability",
}


def normalize_vector(values: Any, count: int, field: str) -> list[float]:
    """Convert a tensor-like vector while rejecting invalid evidence."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    if values is None:
        raise ValueError(f"{field}: missing vector")
    result = [float(value) for value in values]
    if len(result) != count:
        raise ValueError(f"{field}: expected {count} values, got {len(result)}")
    if not all(math.isfinite(value) and value >= 0.0 for value in result):
        raise ValueError(f"{field}: values must be finite and non-negative")
    if not math.isclose(sum(result), 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"{field}: values must sum to one")
    return result


def normalized_entropy(values: list[float]) -> float:
    """Return Shannon entropy normalized to the finite expert count."""
    if len(values) <= 1:
        return 0.0
    return -sum(value * math.log(max(value, 1e-12)) for value in values) / math.log(len(values))


def normalized_gini(values: list[float]) -> float:
    """Return finite-sample-normalized Gini in [0, 1]."""
    ordered = sorted(values)
    count = len(ordered)
    if count <= 1:
        return 0.0
    raw = (2.0 * sum((index + 1) * value for index, value in enumerate(ordered)) / count) - (count + 1.0) / count
    return min(max(raw * count / (count - 1), 0.0), 1.0)


def build_record(
    *,
    run_id: str,
    family: str,
    layer_name: str,
    module_type: str,
    num_experts: int,
    top_k: int,
    expert_usage: Any,
    source: str,
    step: int = 0,
    mode: str = "eval",
    mean_router_probs: Any | None = None,
    mean_topk_weight: Any | None = None,
    aux_loss: dict[str, Any] | None = None,
    tensor_shape: list[int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate one serializable routing record."""
    family = family.lower()
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported family: {family}")
    if num_experts < 1 or not 1 <= top_k <= num_experts:
        raise ValueError("num_experts and top_k are inconsistent")
    usage = normalize_vector(expert_usage, num_experts, "expert_usage")
    probs = (
        normalize_vector(mean_router_probs, num_experts, "mean_router_probs") if mean_router_probs is not None else None
    )
    weights = normalize_vector(mean_topk_weight, top_k, "mean_topk_weight") if mean_topk_weight is not None else None
    aux = aux_loss or {"status": "not_published", "observed": None}
    observed = aux.get("observed")
    if observed is not None and not math.isfinite(float(observed)):
        raise ValueError("aux_loss.observed must be finite or null")
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "step": int(step),
        "mode": mode,
        "family": family,
        "layer_name": layer_name,
        "module_type": module_type,
        "num_experts": int(num_experts),
        "top_k": int(top_k),
        "expert_usage": usage,
        "usage_semantics": USAGE_SEMANTICS[family],
        "mean_router_probs": probs,
        "mean_topk_weight": weights,
        "normalized_entropy": normalized_entropy(usage),
        "normalized_gini": normalized_gini(usage),
        "dominant_share": max(usage),
        "dead_experts": [index for index, value in enumerate(usage) if value <= 0.01],
        "aux_loss": aux,
        "source": source,
        "tensor_shape": tensor_shape,
        "metadata": metadata or {},
    }
    validate_record(record)
    return record


def validate_record(record: dict[str, Any]) -> None:
    """Validate records at every sink boundary."""
    required = {
        "schema_version",
        "run_id",
        "step",
        "mode",
        "family",
        "layer_name",
        "module_type",
        "num_experts",
        "top_k",
        "expert_usage",
        "usage_semantics",
        "aux_loss",
        "source",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {record['schema_version']}")
    family = str(record["family"]).lower()
    if family not in SUPPORTED_FAMILIES or record["usage_semantics"] != USAGE_SEMANTICS[family]:
        raise ValueError("family and usage_semantics disagree")
    normalize_vector(record["expert_usage"], int(record["num_experts"]), "expert_usage")
