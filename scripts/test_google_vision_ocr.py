#!/usr/bin/env python3
"""Backend-only test script for Google Vision OCR."""

import argparse
import sys
from pathlib import Path

import cv2

# Allow running from repo root without installation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neobio.ocr.google_vision_backend import run_google_vision_ocr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Google Vision OCR on a stitched OCR image."
    )
    parser.add_argument("image", help="Path to stitched OCR input image")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: Failed to read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    try:
        result = run_google_vision_ocr(image)
    except Exception as exc:
        print(f"ERROR: OCR call failed: {exc}", file=sys.stderr)
        sys.exit(1)

    raw_text = result.get("raw_text", "")
    items = list(result.get("items", []))

    print("Raw text:")
    print(raw_text)
    print()
    print(f"Number of items: {len(items)}")

    for idx, item in enumerate(items, start=1):
        text = str(item.get("text", ""))
        cx = float(item.get("cx", 0.0))
        cy = float(item.get("cy", 0.0))
        print(f"{idx:03d}. {text} (cx={cx:.1f}, cy={cy:.1f})")


if __name__ == "__main__":
    main()
