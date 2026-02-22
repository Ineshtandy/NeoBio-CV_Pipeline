#!/usr/bin/env python3
"""
Single-image blot detection runner.

Usage:
    PYTHONPATH=src python scripts/run_blot.py path/to/image.jpg [--band-mode mask] [--debug-out path]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neobio.pipelines import run_blot_pipeline
from neobio.utils import draw_blot_debug, default_out_path


def main():
    parser = argparse.ArgumentParser(
        description="Detect blot lanes and band presence in a single image."
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--band-mode",
        default="mask",
        help="Band detection mode (default: mask)",
    )
    parser.add_argument(
        "--debug-out",
        help="Optional path to save debug overlay image",
    )
    parser.add_argument(
        "--debug-dir",
        default="testing output images",
        help="Directory for debug output if --debug-out not specified (default: testing output images)",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print result as JSON",
    )
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: Failed to read image: {args.image}")
        sys.exit(1)
    
    # Run pipeline
    try:
        result = run_blot_pipeline(image, band_mode=args.band_mode)
    except Exception as e:
        print(f"ERROR: Pipeline failed: {e}")
        sys.exit(1)
    
    # Print results
    print(f"Detected lanes: {result['num_lanes']}")
    print(f"Band present: {result['band_present']}")
    
    if args.print_json:
        # Convert tuples to lists for JSON serialization
        result_json = {
            "num_lanes": result["num_lanes"],
            "band_present": result["band_present"],
            "lane_boxes": result["lane_boxes"],
            "roi": result["roi"],
        }
        print(json.dumps(result_json, indent=2))
    
    # Draw and save debug image
    debug_img = draw_blot_debug(
        image,
        result["roi"],
        result["lane_boxes"],
        result["band_present"],
    )
    
    # Determine output path
    if args.debug_out:
        debug_path = args.debug_out
    else:
        debug_path = default_out_path(args.image, args.debug_dir)
    
    cv2.imwrite(debug_path, debug_img)
    print(f"Debug image saved: {debug_path}")


if __name__ == "__main__":
    main()
