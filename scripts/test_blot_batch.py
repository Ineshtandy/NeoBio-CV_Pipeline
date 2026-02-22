#!/usr/bin/env python3
"""
Batch blot detection runner.

Processes all images in a directory, runs pipeline, and saves debug overlays.
Replaces old test_run() function.

Usage:
    PYTHONPATH=src python scripts/test_blot_batch.py [--input-dir path] [--output-dir path] [--band-mode mode]
"""

import argparse
import sys
from pathlib import Path

import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neobio.pipelines import run_blot_pipeline
from neobio.utils import draw_blot_debug, list_images, ensure_dir, default_out_path


def main():
    parser = argparse.ArgumentParser(
        description="Batch process images for blot detection."
    )
    parser.add_argument(
        "--input-dir",
        default="NeoBio Input Images",
        help="Input directory with images (default: NeoBio Input Images)",
    )
    parser.add_argument(
        "--output-dir",
        default="testing output images",
        help="Output directory for debug images (default: testing output images)",
    )
    parser.add_argument(
        "--band-mode",
        default="mask",
        help="Band detection mode (default: mask)",
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    # List images
    image_paths = list_images(args.input_dir)
    
    if not image_paths:
        print(f"No images found in '{args.input_dir}'")
        return
    
    print(f"Found {len(image_paths)} image(s)")
    print("-" * 80)
    
    processed = 0
    failed = 0
    
    for idx, image_path in enumerate(image_paths, 1):
        image_name = Path(image_path).name
        print(f"[{idx}/{len(image_paths)}] Processing: {image_name}")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"  ✗ Failed to read image")
                failed += 1
                continue
            
            # Run pipeline
            result = run_blot_pipeline(image, band_mode=args.band_mode)
            
            # Print results
            num_lanes = result["num_lanes"]
            band_present = result["band_present"]
            print(f"  ✓ Lanes: {num_lanes}, Bands: {band_present}")
            
            # Draw and save debug image
            debug_img = draw_blot_debug(
                image,
                result["roi"],
                result["lane_boxes"],
                result["band_present"],
            )
            
            debug_path = default_out_path(image_path, args.output_dir, suffix="_debug.png")
            cv2.imwrite(debug_path, debug_img)
            processed += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
    
    print("-" * 80)
    print(f"Summary: {processed} processed, {failed} failed (total: {len(image_paths)})")


if __name__ == "__main__":
    main()
