#!/usr/bin/env python3
"""Single-image integrated blot + OCR pipeline runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Allow running from repo root without installation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neobio.pipelines import run_integrated_pipeline
from neobio.utils import default_out_path, ensure_dir


def _to_jsonable(value: Any) -> Any:
    """Convert nested values into JSON-serializable native Python types."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run integrated pipeline: shared preprocessing + OCR + blot."
    )
    parser.add_argument("image", help="Path to input blot image")
    parser.add_argument(
        "--output-json",
        metavar="PATH",
        help="Optional path to save integrated result JSON",
    )
    parser.add_argument(
        "--stitched-out",
        metavar="PATH",
        help="Optional path to save stitched OCR input image",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug artefacts from branch pipelines",
    )
    parser.add_argument(
        "--debug-dir",
        default="testing output images/integrated_pipeline_debug",
        metavar="PATH",
        help="Root directory for debug artefacts",
    )

    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: Failed to read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    try:
        result = run_integrated_pipeline(
            image,
            debug=args.debug,
            debug_dir=args.debug_dir,
        )
    except Exception as exc:
        print(f"ERROR: Integrated pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Identifier: {result['identifier']}")
    print("Reactivity mapping:")
    for label, has_band in result["reactivity"].items():
        print(f"- {label}: {has_band}")

    print(f"ROI: {result['meta']['roi']}")
    print(f"Top labels: {result['ocr']['top_labels']}")
    print(f"Bottom identifier: {result['ocr']['bottom_identifier']}")
    print(f"Band present: {result['blot']['band_present']}")
    print("Merged lanes:")
    for lane in result["lanes"]:
        print(
            f"- idx={lane['lane_index']} label='{lane['label']}' "
            f"band_present={lane['band_present']} lane_box={lane['lane_box']}"
        )

    stitched_image = result["ocr"].get("stitched_image")
    if stitched_image is not None:
        if args.stitched_out:
            stitched_path = args.stitched_out
            ensure_dir(str(Path(stitched_path).parent))
        else:
            stitched_path = default_out_path(
                args.image,
                "testing output images/integrated_pipeline_output",
                suffix="_integrated_stitched.png",
            )
        cv2.imwrite(stitched_path, stitched_image)
        print(f"Stitched OCR input saved: {stitched_path}")

    if args.output_json:
        output_path = args.output_json
        ensure_dir(str(Path(output_path).parent))

        json_result = dict(result)
        if "ocr" in json_result:
            json_ocr = dict(json_result["ocr"])
            json_ocr.pop("stitched_image", None)
            json_result["ocr"] = json_ocr

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(json_result), f, indent=2)
        print(f"Integrated JSON saved: {output_path}")


if __name__ == "__main__":
    main()
