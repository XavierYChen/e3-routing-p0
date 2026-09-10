"""Machine-readable and static-figure sinks for P0 records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import SCHEMA_VERSION, validate_record


class JsonlSink:
    """Append validated records so training loops can stream without retaining history."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, records: list[dict[str, Any]]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            for record in records:
                validate_record(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def write_snapshot(path: str | Path, records: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Write one complete snapshot with a stable envelope."""
    for record in records:
        validate_record(record)
    payload = {"schema_version": SCHEMA_VERSION, "metadata": metadata, "records": records}
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _annotate_heatmap(axis: Any, matrix: Any, image: Any) -> None:
    import numpy as np

    size = 6 if matrix.shape[1] > 8 else 8
    for row, column in np.ndindex(matrix.shape):
        value = matrix[row, column]
        if np.isnan(value):
            continue
        red, green, blue, _ = image.cmap(image.norm(value))
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        axis.text(
            column,
            row,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=size,
            color="white" if luminance < 0.48 else "#172033",
        )


def render_static(records: list[dict[str, Any]], output: str | Path, *, title: str = "E3 P0 routing snapshot") -> Path:
    """Render all present families with values and normalized balance metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for record in records:
        validate_record(record)
    families = [family for family in ("moe", "mot", "latent") if any(r["family"] == family for r in records)]
    if not families:
        raise ValueError("no supported routing records to plot")
    figure, axes = plt.subplots(len(families), 2, figsize=(14, 4.0 * len(families)), squeeze=False)
    for row, family in enumerate(families):
        selected = [record for record in records if record["family"] == family]
        columns = max(record["num_experts"] for record in selected)
        matrix = np.full((len(selected), columns), np.nan)
        for index, record in enumerate(selected):
            matrix[index, : record["num_experts"]] = record["expert_usage"]
        left, right = axes[row]
        image = left.imshow(matrix, vmin=0.0, vmax=max(1.0 / columns, np.nanmax(matrix)), cmap="Blues", aspect="auto")
        semantics = "Top-k selection share" if family == "moe" else "Mean mixture probability"
        left.set_title(f"{family.upper()} — {semantics}")
        left.set_xticks(range(columns))
        left.set_yticks(range(len(selected)), [record["layer_name"] for record in selected])
        left.set_xlabel("Expert index")
        _annotate_heatmap(left, matrix, image)
        figure.colorbar(image, ax=left, fraction=0.025, pad=0.02)
        positions = np.arange(len(selected))
        entropy = right.barh(
            positions - 0.18,
            [record["normalized_entropy"] for record in selected],
            0.36,
            label="Entropy",
        )
        gini = right.barh(
            positions + 0.18,
            [record["normalized_gini"] for record in selected],
            0.36,
            label="Gini",
        )
        for bars in (entropy, gini):
            for bar in bars:
                value = float(bar.get_width())
                right.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
        right.set_yticks(positions, [record["layer_name"] for record in selected])
        right.invert_yaxis()
        right.set_xlim(0.0, 1.08)
        right.set_title("Normalized routing balance (0–1)")
        right.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle(title, fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def render_contract_summary(records: list[dict[str, Any]], output: str | Path) -> Path:
    """Render P0 schema coverage so this stage is visibly distinct from smoke."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for record in records:
        validate_record(record)
    families = [family for family in ("moe", "mot", "latent") if any(r["family"] == family for r in records)]
    checks = {
        "Schema valid": lambda record: record["schema_version"] == SCHEMA_VERSION,
        "Usage normalized": lambda record: abs(sum(record["expert_usage"]) - 1.0) <= 1e-4,
        "Semantics explicit": lambda record: (
            record["usage_semantics"] in {"topk_selection_share", "mean_mixture_probability"}
        ),
        "Source explicit": lambda record: bool(record["source"]),
        "Aux state explicit": lambda record: "status" in record["aux_loss"] and "observed" in record["aux_loss"],
    }
    coverage = np.zeros((len(checks), len(families)))
    counts = []
    for column, family in enumerate(families):
        selected = [record for record in records if record["family"] == family]
        counts.append(len(selected))
        for row, check in enumerate(checks.values()):
            coverage[row, column] = sum(bool(check(record)) for record in selected) / len(selected)

    figure, (left, middle, right) = plt.subplots(
        1,
        3,
        figsize=(15, 4.8),
        gridspec_kw={"width_ratios": [0.8, 1.4, 1.2]},
    )
    bars = left.bar([family.upper() for family in families], counts, color=["#2563eb", "#7c3aed", "#0f766e"])
    left.set_title("Captured routed layers")
    left.set_ylabel("Layer records")
    left.bar_label(bars, labels=[str(count) for count in counts], padding=3, fontsize=11)
    left.set_ylim(0, max(counts) + 1.5)

    image = middle.imshow(coverage, vmin=0.0, vmax=1.0, cmap="Greens", aspect="auto")
    middle.set_title("Validated contract coverage")
    middle.set_xticks(range(len(families)), [family.upper() for family in families])
    middle.set_yticks(range(len(checks)), list(checks))
    for row, column in np.ndindex(coverage.shape):
        value = coverage[row, column]
        middle.text(
            column,
            row,
            f"{value:.0%}",
            ha="center",
            va="center",
            color="white" if value > 0.55 else "#172033",
        )
    figure.colorbar(image, ax=middle, fraction=0.045, pad=0.03, label="Passing records")

    right.axis("off")
    right.set_title("P0 deliverables")
    deliverables = [
        ("Versioned schema", SCHEMA_VERSION),
        ("Reusable collector", "RoutingCollector"),
        ("Streaming sink", "JSONL append"),
        ("Snapshot sink", "JSON envelope"),
        ("Static sink", "annotated PNG"),
        ("Core forward changes", "none"),
    ]
    for row, (name, value) in enumerate(deliverables):
        y = 0.9 - row * 0.14
        right.text(0.02, y, "✓", color="#15803d", fontsize=15, fontweight="bold", va="center")
        right.text(0.10, y, name, fontsize=9.5, fontweight="bold", va="center")
        right.text(0.70, y, value, fontsize=9.5, color="#334155", va="center")
    right.text(
        0.02,
        0.02,
        "Smoke asks: can one snapshot be captured?\n"
        "P0 asks: can every supported family emit the same reusable contract?",
        fontsize=9,
        color="#475569",
    )
    figure.suptitle("E3 P0 · Unified routing observability contract", fontsize=16)
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def render_cross_family_summary(
    records: list[dict[str, Any]], output: str | Path, *, context: str = "one validated routing snapshot"
) -> Path:
    """Summarize comparable normalized metrics across families without hiding semantics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for record in records:
        validate_record(record)
    families = [family for family in ("moe", "mot", "latent") if any(r["family"] == family for r in records)]
    if not families:
        raise ValueError("no supported routing records to summarize")
    metrics = [
        ("Mean normalized entropy", "normalized_entropy", "#2563eb"),
        ("Mean load Gini", "normalized_gini", "#f59e0b"),
        ("Mean dominant share", "dominant_share", "#10b981"),
    ]
    x = np.arange(len(families))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10.8, 6.2))
    for index, (label, key, color) in enumerate(metrics):
        values = [
            float(np.mean([record[key] for record in records if record["family"] == family]))
            for family in families
        ]
        bars = axis.bar(x + (index - 1) * width, values, width, label=label, color=color, alpha=0.9)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
    layer_counts = [sum(record["family"] == family for record in records) for family in families]
    axis.set_xticks(x, [f"{family.upper()}\n(n={count} layers)" for family, count in zip(families, layer_counts)])
    axis.set_ylabel("Metric value (0–1)")
    axis.set_ylim(0.0, 1.12)
    axis.set_title(f"E3 P0 cross-family routing summary\n{context}")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left", ncols=3, fontsize=9)
    figure.text(
        0.5,
        0.015,
        "Layer means. MOE usage is Top-K selection share; MOT/LATENT usage is mean mixture probability.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output
