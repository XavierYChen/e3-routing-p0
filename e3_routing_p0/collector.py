"""Non-invasive forward-hook adapters for three routing families."""

from __future__ import annotations

import math
from typing import Any, Self

from .schema import build_record


def module_family(module: Any) -> str:
    """Resolve the canonical family without changing the module."""
    snapshot = getattr(module, "last_routing_snapshot", {})
    if isinstance(snapshot, dict) and snapshot.get("family"):
        return str(snapshot["family"]).lower()
    kind = getattr(module, "_routing_aux_kind", None)
    if kind:
        return str(kind).lower()
    module_path = type(module).__module__.lower()
    return next((family for family in ("latent", "mot", "moe") if family in module_path), "unknown")


def aux_status(module: Any, family: str) -> dict[str, Any]:
    """Describe auxiliary-loss state without turning missing evidence into zero."""
    snapshot = getattr(module, "last_routing_snapshot", {})
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    balance = float(getattr(module, "balance_loss_coeff", 0.0) or 0.0)
    z_loss = float(getattr(module, "router_z_loss_coeff", 0.0) or 0.0)
    configured = balance > 0.0 or z_loss > 0.0
    if family == "moe":
        status, observed = "not_published_eval", None
    else:
        observed = snapshot.get("aux_loss")
        observed = float(observed) if observed is not None else None
        status = (
            "observed" if observed is not None else ("configured_inactive_eval" if configured else "not_configured")
        )
    if observed is not None and not math.isfinite(observed):
        raise ValueError("non-finite auxiliary loss")
    return {
        "status": status,
        "observed": observed,
        "balance_loss_coeff": balance,
        "router_z_loss_coeff": z_loss,
    }


class RoutingCollector:
    """Collect fresh leaf snapshots during a forward pass and remove every hook."""

    def __init__(
        self,
        model: Any,
        *,
        run_id: str,
        families: tuple[str, ...] = ("moe", "mot", "latent"),
        step: int = 0,
        mode: str = "eval",
    ) -> None:
        requested = {family.lower() for family in families}
        candidates = {
            name: module
            for name, module in model.named_modules()
            if hasattr(module, "last_routing_snapshot") and module_family(module) in requested
        }
        self.modules = {
            name: module
            for name, module in candidates.items()
            if not any(other.startswith(name + ".") for other in candidates)
        }
        if not self.modules:
            raise ValueError(f"unsupported: no routed modules found for {sorted(requested)}")
        self.run_id, self.step, self.mode = run_id, step, mode
        self.records: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        self.previous: dict[str, Any] = {}
        self.pending_moe: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> Self:
        self.previous = {name: module.last_routing_snapshot for name, module in self.modules.items()}
        try:
            for name, module in self.modules.items():
                family = module_family(module)
                if family == "moe":
                    if type(module).__name__ != "OptimizedMOEImproved" or not hasattr(module, "routing"):
                        raise ValueError(f"unsupported MoE adapter: {type(module).__name__}")
                    self.handles.append(module.routing.register_forward_hook(self._moe_router_hook(name, module)))
                self.handles.append(module.register_forward_hook(self._module_hook(name, family)))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def _moe_router_hook(self, name: str, owner: Any):
        def capture(_router: Any, _inputs: Any, output: Any) -> None:
            import torch

            weights, indices = output[:2]
            if weights.ndim != 2 or weights.shape != indices.shape:
                raise ValueError(f"{name}: unsupported top-k layout")
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError(f"{name}: invalid top-k weights")
            if indices.is_floating_point() or (indices < 0).any() or (indices >= owner.num_experts).any():
                raise ValueError(f"{name}: invalid expert indices")
            if not torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]), atol=1e-4):
                raise ValueError(f"{name}: top-k weights do not sum to one")
            counts = torch.bincount(indices.reshape(-1), minlength=owner.num_experts).float()
            dense = weights.new_zeros((weights.shape[0], owner.num_experts))
            dense.scatter_add_(1, indices.long(), weights)
            self.pending_moe[name] = {
                "num_experts": owner.num_experts,
                "top_k": weights.shape[1],
                "expert_usage": counts / counts.sum(),
                "mean_router_probs": dense.mean(0),
                "mean_topk_weight": weights.mean(0),
            }

        return capture

    def _module_hook(self, name: str, family: str):
        def capture(module: Any, _inputs: Any, output: Any) -> None:
            snapshot = self.pending_moe.pop(name, None) if family == "moe" else module.last_routing_snapshot
            if not snapshot or (family != "moe" and snapshot is self.previous[name]):
                raise ValueError(f"{name}: missing or stale routing snapshot")
            diagnostics = snapshot.get("finite_diagnostics", {})
            if snapshot.get("finite") is False or diagnostics.get("all_finite") is False:
                raise ValueError(f"{name}: producer reports non-finite routing")
            self.previous[name] = snapshot
            tensor = output[0] if isinstance(output, tuple) else output
            self.records.append(
                build_record(
                    run_id=self.run_id,
                    family=family,
                    layer_name=name,
                    module_type=type(module).__name__,
                    num_experts=int(snapshot["num_experts"]),
                    top_k=int(snapshot["top_k"]),
                    expert_usage=snapshot.get("expert_usage"),
                    mean_router_probs=snapshot.get("mean_router_probs"),
                    mean_topk_weight=snapshot.get("mean_topk_weight"),
                    aux_loss=aux_status(module, family),
                    source="router_topk_output" if family == "moe" else "last_routing_snapshot",
                    step=self.step,
                    mode=self.mode,
                    tensor_shape=list(tensor.shape) if hasattr(tensor, "shape") else None,
                    metadata={
                        "usage_scope": snapshot.get("usage_scope", "rank_local"),
                        "global_usage_available": bool(snapshot.get("global_usage_available", False)),
                        "dispatch_policy": snapshot.get("dispatch_policy"),
                    },
                )
            )

        return capture

    def __exit__(self, *_args: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending_moe.clear()
