"""Collect a reusable P0 routing snapshot from the locked YOLO-Master checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .collector import RoutingCollector
from .sinks import JsonlSink, render_contract_summary, render_cross_family_summary, render_static, write_snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--yolo-root",
        type=Path,
        default=Path(os.environ.get("E3_YOLO_MASTER_ROOT", PROJECT_ROOT.parent / "YOLO-Master")),
    )
    result.add_argument("--data", default="coco8.yaml")
    result.add_argument("--image", type=Path)
    result.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results" / datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")
    )
    result.add_argument("--families", nargs="+", choices=("moe", "mot", "latent"), default=["moe", "mot", "latent"])
    result.add_argument("--imgsz", type=int, default=320)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="cpu")
    return result


def resolve_image(data: str, override: Path | None) -> tuple[Path, dict[str, str]]:
    from ultralytics.data.utils import check_det_dataset

    dataset = check_det_dataset(data, autodownload=False)
    root = Path(str(dataset["path"])).resolve()
    metadata = {"config": data, "root_name": root.name}
    if override:
        image = override.expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        return image, metadata
    values = dataset.get("val")
    values = values if isinstance(values, (list, tuple)) else [values]
    candidates: list[Path] = []
    for value in values:
        path = Path(str(value))
        if path.is_dir():
            candidates.extend(
                item for item in path.rglob("*") if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
        elif path.is_file():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"no validation images resolved from {data}")
    return min(candidates).resolve(), metadata


def load_batch(path: Path, imgsz: int, device):
    import numpy as np
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.patches import imread

    image = imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    resized = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=image)
    rgb = np.ascontiguousarray(resized[..., ::-1].transpose(2, 0, 1))
    return torch.from_numpy(rgb).unsqueeze(0).to(device=device, dtype=torch.float32).div_(255.0)


def load_model(path: Path, device):
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(str(path), ch=3, verbose=False).to(device).eval().float()
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = True
    return model


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.yolo_root = args.yolo_root.resolve()
    if not (args.yolo_root / "ultralytics").is_dir():
        raise SystemExit(f"YOLO-Master not found: {args.yolo_root}")
    sys.path.insert(0, str(args.yolo_root))
    os.environ["MOE_SNAPSHOT_INTERVAL"] = "1"
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    config_dir = Path(os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "env/runtime")))
    config_dir.mkdir(parents=True, exist_ok=True)
    import torch
    from ultralytics.utils import SETTINGS

    local_datasets = args.yolo_root.parent / "datasets"
    if (local_datasets / "coco8").is_dir():
        SETTINGS.update({"datasets_dir": str(local_datasets)})
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    image, dataset = resolve_image(args.data, args.image)
    batch = load_batch(image, args.imgsz, device)
    run_id = args.output.name
    records: list[dict] = []
    profiles = {
        "moe": "yolo26-master-n.yaml",
        "mot": "yolo26-master-mot-n.yaml",
        "latent": "yolo26-master-latent-n.yaml",
    }
    for family in args.families:
        model_path = args.yolo_root / "ultralytics/cfg/models/26" / profiles[family]
        model = load_model(model_path, device)
        with RoutingCollector(model, run_id=run_id, families=(family,)) as collector, torch.no_grad():
            model(batch)
        records.extend(collector.records)
        print(f"[{family}] captured {len(collector.records)} layers", flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "passed",
        "families": args.families,
        "input": {"image": image.name, "sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "dataset": dataset},
        "runtime": {"device": str(device), "seed": args.seed, "imgsz": args.imgsz, "batch": 1, "dtype": "float32"},
        "yolo_master": {
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=args.yolo_root, capture_output=True, text=True, check=False
            ).stdout.strip(),
            "official_base_commit": "246e79cfe418cfd90f4738bace56b02245dc38f8",
        },
        "model_configs": {
            family: {
                "path": f"ultralytics/cfg/models/26/{profiles[family]}",
                "sha256": hashlib.sha256(
                    (args.yolo_root / "ultralytics/cfg/models/26" / profiles[family]).read_bytes()
                ).hexdigest(),
            }
            for family in args.families
        },
    }
    write_snapshot(args.output / "routing_snapshot.json", records, metadata)
    JsonlSink(args.output / "routing_snapshot.jsonl").write(records)
    render_static(
        records,
        args.output / "routing_snapshot.png",
        title=f"E3 P0 unified routing snapshot — {image.name} — seed {args.seed}",
    )
    render_contract_summary(records, args.output / "p0_contract_coverage.png")
    render_cross_family_summary(
        records,
        args.output / "routing_cross_family.png",
        context=f"{image.name} · {device} · seed={args.seed} · imgsz={args.imgsz}",
    )
    (args.output / "summary.json").write_text(
        json.dumps({"status": "passed", "records": len(records), "families": args.families}, indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts = {}
    for path in sorted(args.output.iterdir()):
        if path.is_file():
            artifacts[path.name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    (args.output / "manifest.sha256.json").write_text(
        json.dumps({"algorithm": "sha256", "files": artifacts}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Result: passed | evidence={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
