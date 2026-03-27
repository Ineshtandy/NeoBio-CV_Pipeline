# NeoBio CV Pipeline

A scalable computer vision pipeline for automated blot detection and lane band presence analysis.

## Overview

This package provides:
- **Lane Detection**: Automatically detects lanes in blot/membrane images
- **Band Detection**: MVP mask-only detector identifies protein bands in each lane
- **OCR Preparation**: Crops and stitches text regions around blot ROI for downstream processing
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

## Running Instructions

### Blot Detection (Lane + Band Analysis)

The simplest workflow: detect lanes and identify which lanes contain protein bands.

**Single Image**

```bash
PYTHONPATH=src python scripts/run_blot.py path/to/image.jpeg
```

Output:
```
Detected lanes: 8
Band present: [True, False, True, False, True, False, False, True]
Debug image saved: testing output images/image_debug.png
```

With JSON output:
```bash
PYTHONPATH=src python scripts/run_blot.py "NeoBio Input Images/sample.jpg" --print-json
```

**Batch Processing**

Process all images in a directory:

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

With custom band mode:
```bash
PYTHONPATH=src python scripts/run_blot.py image.jpg --band-mode mask
```

---

### OCR Preparation (Text Region Extraction)

Extract and stitch text regions above and below the blot for downstream OCR processing. No text extraction performed here—only image preparation.

**Single Image**

```bash
PYTHONPATH=src python scripts/run_ocr_prep.py path/to/image.jpg
```

Output:
```
Stitched image saved: ocr_prep_debug/image_stitched.png
```

**With Debug Output and JSON Metadata**

```bash
PYTHONPATH=src python scripts/run_ocr_prep.py path/to/image.jpg \
  --debug --debug-dir "ocr_prep_debug" \
  --top-extra-bottom-px 5 \
  --bottom-extra-top-px 5 \
  --gap-px 20 \
  --print-json
```

Debug artifacts saved per image:
```
ocr_prep_debug/<image_stem>/
  01_full_image.png
  02_lane_mask.png
  03_top_region.png
  04_bottom_region.png
  05_stitched_ocr_input.png
  06_roi_overlay.png
```

---

### Integrated Pipeline (Recommended: Reactivity Output)

Complete end-to-end workflow using shared preprocessing for both OCR and blot branches, then merges outputs into final reactivity mapping.

**Single Image**

```bash
PYTHONPATH=src python scripts/run_integrated_pipeline.py path/to/image.jpg
```

Output:
```
Identifier: NKX2.1/15721
Reactivity mapping:
- PAX8: True
- NAPSIN A: False
...
```

**With Debug + JSON Export**

```bash
PYTHONPATH=src python scripts/run_integrated_pipeline.py path/to/image.jpg \
  --debug --debug-dir "testing output images/integrated_pipeline_debug" \
  --output-json "testing output images/integrated_pipeline_output/result_integrated.json" \
  --stitched-out "testing output images/integrated_pipeline_output/result_stitched.png"
```

---

### OCR-Only Pipeline (Text Extraction)

Complete end-to-end OCR workflow: image analysis → region extraction → Google Vision OCR → text post-processing.

**Single Image**

```bash
PYTHONPATH=src python scripts/run_ocr_pipeline.py path/to/image.jpg
```

**With Full Options**

```bash
PYTHONPATH=src python scripts/run_ocr_pipeline.py path/to/image.jpg \
  --top-extra-bottom-px 6 \
  --bottom-extra-top-px 6 \
  --top-rotation-deg -52.2 \
  --stitch-gap-px 20 \
  --debug --debug-dir "testing output images/ocr_pipeline_debug"
```

**Backend-Only OCR Test**

Test OCR processing on a pre-stitched image:

```bash
PYTHONPATH=src python scripts/test_google_vision_ocr.py path/to/stitched.png
```

---

## Project Structure

```
src/neobio/
├── __init__.py
├── blot/
│   ├── __init__.py
│   ├── lane_mask.py              # Lane detection and ROI extraction
│   ├── band_detection.py         # MVP mask-only band detector
│   └── band_detection_legacy.py  # Legacy reference implementations (not used by default)
├── ocr/
│   ├── __init__.py
│   ├── region_stitching.py       # Region bounds, cropping, rotation, and stitching helpers
│   ├── google_vision_backend.py  # Google Vision API calls and response parsing
│   └── ocr_postprocess.py        # Text post-processing (split, group, sort, join)
├── pipelines/
│   ├── __init__.py
│   ├── blot_pipeline.py          # Main blot lane/band orchestrator
│   ├── ocr_prep_pipeline.py      # OCR region extraction and stitching (no text analysis)
│   ├── ocr_pipeline.py           # Full OCR orchestrator (prep → Vision → post-process)
│   └── shared_preprocessing.py   # Common preprocessing utilities
└── utils/
    ├── __init__.py
    ├── debug_draw.py             # Visualization and debug image generation
    └── io.py                     # File I/O helpers

scripts/
├── run_blot.py                   # Single-image blot lane/band detection
├── test_blot_batch.py            # Batch lane/band detection for directories
├── run_ocr_prep.py               # Single-image OCR region extraction
├── run_ocr_pipeline.py           # Single-image OCR-only extraction pipeline
├── run_integrated_pipeline.py    # Single-image integrated OCR + blot + reactivity pipeline
└── test_google_vision_ocr.py     # Backend-only OCR test (Vision API call + parsing)
```

## Python API

### Blot Pipeline

**Basic Usage**

```python
import sys
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

**Output Format**

```python
{
  "num_lanes": 8,
  "band_present": [True, False, True, ...],
  "lane_boxes": [(x1, y1, x2, y2), ...],
  "roi": (x1, y1, x2, y2)
}
```

---

### OCR Preparation Pipeline

Extract and stitch text regions around the blot ROI without performing text analysis.

```python
import cv2
from neobio.pipelines import run_ocr_prep_pipeline

image = cv2.imread("path/to/image.jpg")
result = run_ocr_prep_pipeline(
    image,
    top_extra_bottom_px=5,    # Extend top strip 5 px into ROI
    bottom_extra_top_px=5,    # Extend bottom strip 5 px into ROI
    gap_px=20,                # White separator between strips
    debug=True,
    debug_dir="ocr_prep_debug",
    input_path="path/to/image.jpg",
)

stitched = result["stitched_image"]  # numpy.ndarray ready for OCR
roi      = result["roi"]             # (x1, y1, x2, y2)
```

**Output Schema**

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

---

### Integrated Pipeline

Run OCR and blot branches with one shared preprocessing context, then merge into a final reactivity view.

```python
import cv2
from neobio.pipelines import run_integrated_pipeline

image = cv2.imread("path/to/image.jpg")
result = run_integrated_pipeline(
    image,
    debug=True,
    debug_dir="testing output images/integrated_pipeline_debug",
)
```

**Output Fields**

```python
{
  "identifier": "...",
  "reactivity": {"PAX8": True, "NAPSIN A": False, ...},
  "ocr": {
    "top_labels": ["...", "..."],
    "bottom_identifier": "...",
    "top_items": [...],
    "bottom_items": [...],
    "raw_text": "...",
    "stitch_boundary_y": int,
    "top_region_bounds": (x1, y1, x2, y2),
    "bottom_region_bounds": (x1, y1, x2, y2),
    "stitched_image": np.ndarray,
  },
  "blot": {
    "num_lanes": int,
    "lane_boxes": [(x1, y1, x2, y2), ...],
    "band_present": [True, False, ...],
  },
  "lanes": [...],
  "meta": {...}
}
```

---

### OCR-Only Pipeline

Complete text extraction from original image to processed output.

```python
import cv2
from neobio.pipelines import run_ocr_pipeline

image = cv2.imread("path/to/image.jpg")
result = run_ocr_pipeline(
    image,
    top_extra_bottom_px=6,
    bottom_extra_top_px=6,
    top_rotation_deg=-52.2,
    stitch_gap_px=20,
    debug=True,
    debug_dir="testing output images/ocr_pipeline_debug",
)
```

**Output Fields**

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

**Debug Artifacts**

```
ocr_pipeline_debug/<timestamp>/
  01_full_image.png
  02_top_crop.png
  03_top_crop_rotated.png
  04_bottom_crop.png
  05_stitched_ocr_input.png
```

---

## Understanding the Pipelines

### Lane Detection Algorithm

The lane detection pipeline processes images in four steps:

1. **Mask Creation** (`build_lane_mask`): Converts image to binary mask (lanes white, separators black)
2. **ROI Extraction** (`roi_from_mask`): Finds bounding box of blot region
3. **Separator Detection** (`find_vertical_separators`): Detects lane dividing lines
4. **Lane Boxes** (`lanes_from_separators`): Generates individual lane bounding boxes

### Band Detection Algorithm

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

**Configuration Parameters**

For detailed band detector configuration parameters (thresholds, crop fractions, kernel sizes), see [DESIGN.md](DESIGN.md#band-detector-configuration).

---

## Development

### Testing

```bash
# Single image with JSON output
PYTHONPATH=src python scripts/run_blot.py "NeoBio Input Images/sample.jpg" --print-json

# Batch processing all images
PYTHONPATH=src python scripts/test_blot_batch.py

# Custom band detection mode
PYTHONPATH=src python scripts/run_blot.py image.jpg --band-mode mask
```

### Adding New Band Detectors

1. Implement detector function in `src/neobio/blot/band_detection.py`:
   ```python
   def my_detector(lane_mask: np.ndarray) -> bool:
       # Detection logic
       return True or False
   ```

2. Register in `DETECTORS` dictionary:
   ```python
   DETECTORS["my_mode"] = my_detector
   ```

3. Use via CLI or API:
   ```bash
   PYTHONPATH=src python scripts/run_blot.py image.jpg --band-mode my_mode
   ```

---

## Notes

- Temporary debug artifacts (e.g., `mask_post_morph.png`) are in `.gitignore`
- Input folder `NeoBio Input Images/` is ignored by git
- Output folder `testing output images/` contains debug results and test outputs
- For OCR and integrated pipelines: Set up Google Cloud credentials before running (`GOOGLE_APPLICATION_CREDENTIALS` environment variable)
- OCR backend raises clear errors for image-encode failures and API errors

## Related Documentation

- [DESIGN.md](DESIGN.md) - Detailed architecture and design decisions