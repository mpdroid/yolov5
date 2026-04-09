#!/usr/bin/env python3
"""Export Ultralytics checkpoints (e.g. YOLOv5n6) to native Core ML output."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path


def _bundle_filename(exported: Path, output_name: str | None) -> tuple[str, str | None]:
    """Return final bundle filename (with suffix) or (empty, error message)."""
    if output_name is None:
        return exported.name, None
    stem = output_name.strip()
    if not stem:
        return "", "Empty --output-name"
    if "/" in stem or "\\" in stem or ".." in stem:
        return "", "Invalid --output-name (must be a plain stem, no path segments)"
    return f"{stem}{exported.suffix}", None


def main() -> int:
    backend_dir = Path(__file__).resolve().parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    parser = argparse.ArgumentParser(description="Native Core ML export for Ultralytics .pt checkpoints.")
    parser.add_argument("--weights", required=True, help="Path to best.pt (or other Ultralytics .pt)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for the final native .mlpackage/.mlmodel",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size (default 640)")
    parser.add_argument("--device", default=None, help="e.g. cpu, mps, 0 (default: exporter default)")
    parser.add_argument("--half", action="store_true", help="FP16 weights where supported")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true", help="Variable H×W")
    parser.add_argument("--nms", action="store_true", help="Fuse NMS into export when supported")
    parser.add_argument(
        "--show-coreml-warnings",
        action="store_true",
        help="Show coremltools MIL RuntimeWarnings (e.g. overflow in cast during range analysis; usually benign)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Final bundle base name without extension (e.g. Y5 → Y5.mlpackage); default: Ultralytics output name",
    )
    args = parser.parse_args()

    from ultralytics import YOLO
    from ultralytics.engine.exporter import Exporter

    weights = Path(args.weights).resolve()
    if not weights.is_file():
        print(f"Weights not found: {weights}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(weights))
    overrides = {
        "format": "coreml",
        "mode": "export",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "half": args.half,
        "nms": args.nms,
        "device": args.device,
        "dynamic": args.dynamic,
        "verbose": True,
    }

    # coremltools often emits RuntimeWarning: overflow encountered in cast during MIL passes; not a failed export.
    if not args.show_coreml_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"^overflow encountered in cast",
                category=RuntimeWarning,
            )
            path_str = Exporter(overrides=overrides, _callbacks=yolo.callbacks)(model=yolo.model)
    else:
        path_str = Exporter(overrides=overrides, _callbacks=yolo.callbacks)(model=yolo.model)
    exported = Path(path_str)

    fname, fname_err = _bundle_filename(exported, args.output_name)
    if fname_err:
        print(fname_err, file=sys.stderr)
        return 1

    import shutil

    if exported.suffix == ".mlpackage":
        dest = out_dir / fname
        if dest.exists():
            shutil.rmtree(dest)
        if exported.resolve() != dest.resolve() and exported.is_dir():
            shutil.copytree(exported, dest)
            shutil.rmtree(exported)
        print(dest)
    elif exported.suffix == ".mlmodel":
        dest = out_dir / fname
        if exported.resolve() != dest.resolve() and exported.is_file():
            shutil.copy2(exported, dest)
            exported.unlink()
        print(dest)
    else:
        print(f"Unexpected export path: {exported}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
