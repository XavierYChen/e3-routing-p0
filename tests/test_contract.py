"""P0 schema and collector contract tests."""

import json

import pytest

from e3_routing_p0 import JsonlSink, RoutingCollector, build_record, validate_record


def record(**overrides):
    values = {
        "run_id": "test",
        "family": "mot",
        "layer_name": "model.1",
        "module_type": "FakeMoT",
        "num_experts": 2,
        "top_k": 1,
        "expert_usage": [0.25, 0.75],
        "source": "last_routing_snapshot",
    }
    values.update(overrides)
    return build_record(**values)


def test_three_families_have_explicit_semantics():
    moe = record(family="moe")
    mot = record(family="mot")
    latent = record(family="latent")
    assert moe["usage_semantics"] == "topk_selection_share"
    assert mot["usage_semantics"] == latent["usage_semantics"] == "mean_mixture_probability"


@pytest.mark.parametrize("usage", [[0.4, 0.4], [-0.1, 1.1], [float("nan"), 1.0], [1.0]])
def test_invalid_evidence_is_rejected(usage):
    with pytest.raises(ValueError):
        record(expert_usage=usage)


def test_float16_scale_roundoff_is_normalized():
    value = record(num_experts=3, top_k=2, expert_usage=[0.33325, 0.33325, 0.33325])
    assert sum(value["expert_usage"]) == pytest.approx(1.0, abs=1e-12)


def test_material_probability_error_remains_rejected():
    with pytest.raises(ValueError, match="sum to one"):
        record(num_experts=3, top_k=2, expert_usage=[0.32, 0.32, 0.32])


def test_schema_rejects_semantic_drift():
    value = record()
    value["usage_semantics"] = "hit_rate"
    with pytest.raises(ValueError):
        validate_record(value)


def test_jsonl_sink_is_append_only_and_validated(tmp_path):
    path = tmp_path / "records.jsonl"
    sink = JsonlSink(path)
    sink.write([record(layer_name="first")])
    sink.write([record(layer_name="second")])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["layer_name"] for row in rows] == ["first", "second"]


def test_native_snapshot_is_fresh_and_hooks_are_removed():
    torch = pytest.importorskip("torch")

    class Native(torch.nn.Module):
        _routing_aux_kind = "mot"

        def __init__(self):
            super().__init__()
            self.last_routing_snapshot = {}

        def forward(self, value):
            self.last_routing_snapshot = {
                "family": "mot",
                "num_experts": 2,
                "top_k": 1,
                "expert_usage": [0.5, 0.5],
            }
            return value

    module = Native()
    model = torch.nn.Sequential(module)
    with RoutingCollector(model, run_id="test", families=("mot",)) as collector:
        model(torch.zeros(1, 2))
    assert len(collector.records) == 1
    assert not module._forward_hooks


def test_exception_path_removes_hooks():
    torch = pytest.importorskip("torch")

    class Broken(torch.nn.Module):
        _routing_aux_kind = "latent"

        def __init__(self):
            super().__init__()
            self.last_routing_snapshot = {}

        def forward(self, _value):
            raise RuntimeError("expected")

    module = Broken()
    model = torch.nn.Sequential(module)
    with pytest.raises(RuntimeError), RoutingCollector(model, run_id="test", families=("latent",)):
        model(torch.zeros(1))
    assert not module._forward_hooks
