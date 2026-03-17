# NeoBio CV Pipeline

A scalable computer vision pipeline for automated blot detection and lane band presence analysis.

## Overview

This package provides:
- **Lane Detection**: Automatically detects lanes in blot/membrane images
- **Band Detection**: MVP mask-only detector identifies protein bands in each lane
- **OCR Preparation**: Crops/stitches top and bottom text regions around blot ROI
- **OCR Extraction Pipeline**: Runs Google Vision OCR and post-processes labels/identifiers
- **Batch Processing**: Process entire image directories with consistent output
- **Debug Visualization**: Generate annotated debug images for inspection
- **Modular Architecture**: Clean separation of concerns for easy extension

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Dependencies

```bash
pip install opencv-python numpy google-cloud-vision
```

## Quick Start

### Single Image Processing

Detect lanes and bands in a single image:

```bash
PYTHONPATH=src python scripts/run_blot.py path/to/image.jpeg
```

Output:
```
Detected lanes: 8
Band present: [True, False, True, False, True, False, False, True]
Debug image saved: testing output images/image_debug.png
```

### Batch Processing

Process all images in a directory (replaces old `test_run()`):

```bash
PYTHONPATH=src python scripts/test_blot_batch.py \
  --input-dir "NeoBio Input Images" \
  --output-dir "testing output images"
```

Output:
```
Found 12 image(s)
[1/12] Processing: sample_001.jpg
  ✓ Lanes: 8, Bands: [True, False, True, ...]
...
Summary: 12 processed, 0 failed (total: 12)
```

## Python API

### Basic Usage

```python
import sys
from pathlib import Path
sys.path.insert(0, "src")

import cv2
from neobio.pipelines import run_blot_pipeline
from neobio.utils import draw_blot_debug

# Load image
image = cv2.imread("path/to/image.jpg")

# Run pipeline
result = run_blot_pipeline(image, band_mode="mask")

# Access results
num_lanes = result["num_lanes"]
band_present = result["band_present"]
lane_boxes = result["lane_boxes"]
roi = result["roi"]

# Draw debug overlay
debug_img = draw_blot_debug(image, roi, lane_boxes, band_present)
cv2.imwrite("debug.png", debug_img)
```

### Output Format

```python
{
  "num_lanes": 8,
  "band_present": [True, False, True, ...],
  "lane_boxes": [(x1, y1, x2, y2), ...],
  "roi": (x1, y1, x2, y2)
}
```

## Configuration

### Band Detector Parameters

The MVP detector `has_band_mask()` supports customizable parameters:

- `top_crop_frac` (float, default: 0.15): Crop fraction from top
- `bottom_crop_frac` (float, default: 0.15): Crop fraction from bottom
- `edge_margin_px` (int, default: 6): Margin from edges (pixels)
- `smooth_kernel_h` (int, default: 9): Smoothing kernel height
- `peak_thr` (float, default: 0.12): Peak threshold
- `run_thr` (float, default: 0.08): Run-length threshold
- `min_run` (int, default: 10): Minimum run length

## Project Structure

```
src/neobio/
├── __init__.py
├── blot/
│   ├── __init__.py
│   ├── lane_mask.py              # Lane detection, ROI extraction
│   ├── band_detection.py         # MVP mask-only band detector
│   └── band_detection_legacy.py  # Legacy reference implementations
├── ocr/
│   ├── __init__.py
│   ├── region_stitching.py       # OCR bounds/crop/rotate/stitch helpers
│   ├── google_vision_backend.py  # Google Vision OCR call + response parsing
│   └── ocr_postprocess.py        # OCR item split/group/join helpers
├── pipelines/
│   ├── __init__.py
│   ├── blot_pipeline.py          # Main blot orchestrator
│   ├── ocr_prep_pipeline.py      # Stitched OCR image generation only
│   └── ocr_pipeline.py           # OCR prep + Vision + postprocessing
└── utils/
    ├── __init__.py
    ├── debug_draw.py             # Visualization
    └── io.py                     # I/O helpers

scripts/
├── run_blot.py                   # Single-image blot runner
├── test_blot_batch.py            # Batch blot processor
├── run_ocr_prep.py               # Single-image OCR prep runner
├── run_ocr_pipeline.py           # Single-image OCR extraction runner
└── test_google_vision_ocr.py     # Backend-only OCR test runner
```

## Lane Detection Pipeline

1. **Mask Creation** (`build_lane_mask`): Converts image to binary mask (lanes white, separators black)
2. **ROI Extraction** (`roi_from_mask`): Finds bounding box of blot region
3. **Separator Detection** (`find_vertical_separators`): Detects lane dividing lines
4. **Lane Boxes** (`lanes_from_separators`): Generates individual lane bounding boxes

## Band Detection

**MVP Strategy**: Mask-only detection

- Analyzes binary lane mask (no grayscale)
- Crops region: removes top/bottom and edge margins
- Computes row-score: fraction of black pixels per row
- Decision: peak + run-length thresholds

**Algorithm**:
```
1. Crop lane mask (top/bottom %, left/right px)
2. Count black pixels per row → row_score
3. Smooth row scores (morphological blur)
4. If (max(row_score) > peak_thr) AND (longest_true_run(row_score > run_thr) >= min_run):
     return True (band detected)
5. Else: return False (white/no band)
```

## Development

### Testing

```bash
# Single image with JSON output
PYTHONPATH=src python scripts/run_blot.py "NeoBio Input Images/sample.jpg" --print-json

# Batch with all images
PYTHONPATH=src python scripts/test_blot_batch.py

# Custom band mode
PYTHONPATH=src python scripts/run_blot.py image.jpg --band-mode mask
```

### Adding New Detectors

1. Implement function in `src/neobio/blot/band_detection.py`:
   ```python
   def my_detector(lane_mask: np.ndarray) -> bool:
       # Detection logic
       return True or False
   ```

2. Register in `DETECTORS`:
   ```python
   DETECTORS["my_mode"] = my_detector
   ```

3. Use via CLI or API:
   ```bash
   PYTHONPATH=src python scripts/run_blot.py image.jpg --band-mode my_mode
   ```

## Legacy Code

- Original implementation: `old_blot_detect.py` (preserved for reference)
- Legacy detectors: `src/neobio/blot/band_detection_legacy.py` (not used by default)

## Notes

- Temporary debug artifacts (e.g., `mask_post_morph.png`) are in `.gitignore`
- Input folder `NeoBio Input Images/` is ignored by git
- Output folder `testing output images/` contains debug results

## OCR Preparation Pipeline

Produces a stitched image of the regions **above** and **below** the blot ROI,
ready for downstream OCR processing.  Text extraction is not performed here.

### Single Image

```bash
PYTHONPATH=src python scripts/run_ocr_prep.py path/to/image.jpg
```

Output:
```
Stitched image saved: ocr_prep_debug/image_stitched.png
```

With debug artefacts and JSON metadata:
```bash
PYTHONPATH=src python scripts/run_ocr_prep.py path/to/image.jpg \
  --debug --debug-dir "ocr_prep_debug" \
  --top-extra-bottom-px 5 \
  --bottom-extra-top-px 5 \
  --gap-px 20 \
  --print-json
```

JSON output schema:
```json
{
  "roi": [x1, y1, x2, y2],
  "top_region_bounds": [x1, y1, x2, y2],
  "bottom_region_bounds": [x1, y1, x2, y2],
  "top_region_shape": [h, w, c],
  "bottom_region_shape": [h, w, c],
  "stitched_shape": [h, w, c],
  "stitched_path": "ocr_prep_debug/image_stitched.png",
  "debug_dir": "ocr_prep_debug/image"
}
```

Debug artefacts saved per image (when `--debug` is set):
```
ocr_prep_debug/<image_stem>/
  01_full_image.png
  02_lane_mask.png
  03_top_region.png
  04_bottom_region.png
  05_stitched_ocr_input.png
  06_roi_overlay.png
```

### Python API

```python
import cv2
from neobio.pipelines import run_ocr_prep_pipeline

image = cv2.imread("path/to/image.jpg")
result = run_ocr_prep_pipeline(
    image,
    top_extra_bottom_px=5,    # extend top strip 5 px into ROI
    bottom_extra_top_px=5,    # extend bottom strip 5 px into ROI
    gap_px=20,                # white separator between strips
    debug=True,
    debug_dir="ocr_prep_debug",
    input_path="path/to/image.jpg",
)

stitched = result["stitched_image"]  # numpy.ndarray ready for OCR
roi      = result["roi"]             # (x1, y1, x2, y2)
```

### Project Structure (updated)

```
src/neobio/
├── blot/           # Lane mask + ROI extraction (unchanged)
├── ocr/            # OCR helpers
│   ├── __init__.py
│   ├── region_stitching.py       # bounds helpers, crop_region, rotate_image_bound, stitch_regions
│   ├── google_vision_backend.py  # run_google_vision_ocr + annotation normalization
│   └── ocr_postprocess.py        # split/group/sort/join post-processing helpers
├── pipelines/
│   ├── blot_pipeline.py      # unchanged
│   ├── ocr_prep_pipeline.py  # OCR prep orchestrator (no OCR text extraction)
│   └── ocr_pipeline.py       # Full OCR orchestrator
└── utils/          # I/O and debug draw helpers (unchanged)

scripts/
├── run_blot.py         # unchanged
├── run_ocr_prep.py     # OCR prep CLI
├── run_ocr_pipeline.py # Full OCR pipeline CLI
├── test_google_vision_ocr.py # Backend-only OCR CLI
└── test_blot_batch.py  # unchanged
```

## OCR Text Extraction Pipeline

Runs complete OCR flow from original image to text outputs.

### Single Image

```bash
PYTHONPATH=src python scripts/run_ocr_pipeline.py path/to/image.jpg
```

Common options:

```bash
PYTHONPATH=src python scripts/run_ocr_pipeline.py path/to/image.jpg \
  --top-extra-bottom-px 6 \
  --bottom-extra-top-px 6 \
  --top-rotation-deg -52.2 \
  --stitch-gap-px 20 \
  --debug --debug-dir "testing output images/ocr_pipeline_debug"
```

Backend-only OCR test from a stitched image:

```bash
PYTHONPATH=src python scripts/test_google_vision_ocr.py path/to/stitched.png
```

Output fields from `run_ocr_pipeline(...)`:

```python
{
  "roi": (x1, y1, x2, y2),
  "top_region_bounds": (x1, y1, x2, y2),
  "bottom_region_bounds": (x1, y1, x2, y2),
  "stitch_boundary_y": int,
  "top_labels": ["...", "..."],
  "bottom_identifier": "...",
  "top_items": [...],
  "bottom_items": [...],
  "raw_text": "...",
  "stitched_image": np.ndarray,
}
```

Debug artefacts (when `--debug` is set):

```
ocr_pipeline_debug/<timestamp>/
  01_full_image.png
  02_top_crop.png
  03_top_crop_rotated.png
  04_bottom_crop.png
  05_stitched_ocr_input.png
```

Google Vision notes:
- Set up credentials before running OCR (`GOOGLE_APPLICATION_CREDENTIALS`).
- OCR backend raises clear errors for image-encode failures and API errors.

## Related Documentation

- [DESIGN.md](DESIGN.md) - Detailed architecture and design decisions