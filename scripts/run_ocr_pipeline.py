#!/usr/bin/env python3
"""Single-image OCR pipeline runner."""

import argparse
import sys
from pathlib import Path

import cv2

# Allow running from repo root without installation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neobio.pipelines import run_ocr_pipeline
from neobio.utils import default_out_path, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OCR pipeline: prep + Google Vision + postprocess."
    )
    parser.add_argument("image", help="Path to input blot image")
    parser.add_argument(
        "--top-extra-bottom-px",
        type=int,
        default=6,
        help="Pixels to extend top region downward into ROI (default: 0)",
    )
    parser.add_argument(
        "--bottom-extra-top-px",
        type=int,
        default=6,
        help="Pixels to extend bottom region upward into ROI (default: 0)",
    )
    parser.add_argument(
        "--top-rotation-deg",
        type=float,
        default=-52.2,
        help="Rotation angle for top OCR crop in degrees (default: -52.5)",
    )
    parser.add_argument(
        "--stitch-gap-px",
        type=int,
        default=20,
        help="Separator gap between top and bottom crops (default: 20)",
    )
    parser.add_argument(
        "--stitched-out",
        metavar="PATH",
        help="Where to save stitched OCR input image (optional)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug artefacts",
    )
    parser.add_argument(
        "--debug-dir",
        default="testing output images/ocr_pipeline_debug",
        metavar="PATH",
        help="Root directory for debug artefacts",
    )

    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: Failed to read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    try:
        result = run_ocr_pipeline(
            image,
            top_extra_bottom_px=args.top_extra_bottom_px,
            bottom_extra_top_px=args.bottom_extra_top_px,
            top_rotation_deg=args.top_rotation_deg,
            stitch_gap_px=args.stitch_gap_px,
            debug=args.debug,
            debug_dir=args.debug_dir,
        )
    except Exception as exc:
        print(f"ERROR: OCR pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.stitched_out:
        stitched_path = args.stitched_out
        ensure_dir(str(Path(stitched_path).parent))
    else:
        stitched_path = default_out_path(
            args.image,
            "testing output images/ocr_pipeline_output",
            suffix="_ocr_stitched.png",
        )

    cv2.imwrite(stitched_path, result["stitched_image"])

    print(f"ROI: {result['roi']}")
    print(f"Top region bounds: {result['top_region_bounds']}")
    print(f"Bottom region bounds: {result['bottom_region_bounds']}")
    print(f"Stitch boundary y: {result['stitch_boundary_y']}")
    print("Top labels:")
    for label in result["top_labels"]:
        print(f"- {label}")
    print(f"Bottom identifier: {result['bottom_identifier']}")
    print(f"Stitched OCR input saved: {stitched_path}")


if __name__ == "__main__":
    main()
