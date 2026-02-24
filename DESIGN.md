# NeoBio CV Pipeline — Design Document

High-level architecture and design decisions for the blot detection pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Module Responsibilities](#module-responsibilities)
4. [Data Flow](#data-flow)
5. [Key Algorithms](#key-algorithms)
6. [Detector Registry](#detector-registry)
7. [Future Features](#future-features)
8. [Design Decisions](#design-decisions)

---

## Overview

**Purpose**: Automated detection of protein bands in Western blot images.

**Core Stages**:
1. Lane mask creation (binary segmentation)
2. ROI extraction (bounding box)
3. Lane separator detection (vertical lines)
4. Lane box generation (individual lane regions)
5. Per-lane band detection (binary classification)

**Output**:
```python
{
    "num_lanes": int,
    "band_present": list[bool],
    "lane_boxes": list[tuple],
    "roi": tuple
}
```

---

## Architecture

### Package Structure

```
src/neobio/
├── blot/                      # Core blot detection modules
│   ├── lane_mask.py           # Lane mask & ROI utilities
│   ├── band_detection.py      # MVP detector (mask-only)
│   └── band_detection_legacy.py # Legacy reference
├── pipelines/                 # Pipeline orchestrators
│   └── blot_pipeline.py       # Main entry point
└── utils/                     # Shared utilities
    ├── debug_draw.py          # Visualization
    └── io.py                  # File I/O helpers
```

### Layered Design

```
┌─────────────────────────────────────────┐
│         User Scripts (CLI)              │
│   run_blot.py / test_blot_batch.py      │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│       Pipeline Orchestrator             │
│      blot_pipeline.py                   │
└──────────────────┬──────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼────┐  ┌──────▼─────┐  ┌────▼──────┐
│Lane    │  │Band        │  │Utilities  │
│Mask    │  │Detection   │  │(Draw/IO)  │
└────────┘  └────────────┘  └───────────┘
```

---

## Module Responsibilities

### `src/neobio/blot/lane_mask.py`

**Purpose**: Lane geometry and binary mask utilities.

**Key Functions**:
- `auto_grey_range(gray)` → `(low, high)`: Adaptive grey-value detection
- `build_lane_mask(image, debug, debug_out_path)` → `mask`: Binary segmentation
- `roi_from_mask(mask)` → `(x1, y1, x2, y2)`: ROI bounding box
- `find_vertical_separators(mask_roi)` → `[x_coords]`: Separator detection
- `lanes_from_separators(roi_bbox, sep_xs, pad)` → `[(x1, y1, x2, y2), ...]`: Lane boxes

**Design Notes**:
- All functions are **pure** (no I/O side effects unless `debug=True`)
- Coordinates are in **full image space** (except `mask_roi` which is ROI-relative)
- Returns uint8 mask (0/255) for compatibility with `cv2` operations

### `src/neobio/blot/band_detection.py` (MVP)

**Purpose**: Official band detector using mask-only approach.

**Key Functions**:
- `longest_true_run(b)` → `int`: Utility for run-length analysis
- `has_band_from_row_score(row_score_smooth, peak_thr, run_thr, min_run)` → `bool`: Decision logic
- `has_band_mask(lane_mask, ...)` → `bool`: MVP detector signature

**Detector Registry**:
```python
DETECTORS = {
    "mask": has_band_mask,
}
```

**Design Notes**:
- Detector signature: `(lane_mask: np.ndarray) -> bool`
- All parameters are tunable (easy MVP tuning)
- Decision rule: `peak_ok AND run_ok`

### `src/neobio/blot/band_detection_legacy.py`

**Purpose**: Reference implementations (not used by default).

**Includes**:
- `lane_has_band_from_mask()`: Original mask detector (with width check)
- `lane_has_band_from_row_score()`: Row-score decision

**Design Notes**:
- Preserved for comparison and fallback
- Not exposed in main `DETECTORS` registry
- Can be imported for research or re-enabling

### `src/neobio/pipelines/blot_pipeline.py`

**Purpose**: Main orchestrator that "weaves together" all stages.

**Key Function**:
```python
def run_blot_pipeline(image_bgr, band_mode="mask") -> dict:
```

**Orchestration Steps**:
1. Build lane mask
2. Extract ROI from mask
3. Detect vertical separators
4. Generate lane boxes
5. For each lane box:
   - Extract lane mask crop
   - Apply detector
6. Return aggregated result

**Design Notes**:
- Single entry point for all pipeline operations
- Detects band_mode early and validates
- Fallback: if no lanes detected, uses entire ROI as single lane

### `src/neobio/utils/debug_draw.py`

**Purpose**: Visualization utilities.

**Key Function**:
```python
def draw_blot_debug(image_bgr, roi, lane_boxes, band_present) -> debug_image:
```

**Rendering**:
- Green ROI box
- Green lane boxes (if band detected)
- Red lane boxes (if white/no band)
- Text labels: `"{i}: BAND"` or `"{i}: WHITE"`

### `src/neobio/utils/io.py`

**Purpose**: File I/O helpers.

**Key Functions**:
- `list_images(input_dir)` → `[paths]`: Find image files
- `ensure_dir(path)`: Create directory tree
- `default_out_path(input_path, out_dir, suffix)` → `path`: Generate output filename

---

## Data Flow

### Single Image Processing

```
Image (BGR)
    │
    ├→ [build_lane_mask] → Mask (uint8)
    │                        │
    │                        ├→ [roi_from_mask] → ROI bbox
    │                        │
    │                        ├→ [find_vertical_separators] → Sep X-coords
    │                        │
    │                        └→ [lanes_from_separators] → Lane boxes
    │
    ├→ For each lane box:
    │    Lane crop (from mask)
    │        │
    │        └→ [detector] → bool (band present?)
    │
    └→ Aggregate results:
         num_lanes, band_present[], lane_boxes, roi
```

### Batch Processing

```
Images folder
    │
    └→ For each image:
         ├→ [load image]
         ├→ [run_blot_pipeline]
         ├→ [draw_blot_debug] → debug overlay
         └→ [save debug image]
```

---

## Key Algorithms

### Lane Mask Creation

**Steps**:
1. Convert BGR → Grayscale
2. Auto-detect grey range (percentile-based)
3. `cv2.inRange(gray, low, high)` → binary mask
4. Morphology:
   - Open (remove specks): white → black → white
   - Strengthen separators (black → white → black):
     - Invert mask
     - Close (fill vertical gaps)
     - Dilate (thicken)
     - Invert back

**Result**: Lanes white (255), separators/background black (0)

### Separator Detection

**Algorithm**:
1. Invert ROI mask (separators now white)
2. Morphological OPEN with tall vertical kernel (3×k_h)
3. Find contours
4. Filter by height (≥80% of ROI) and width (≤12 px)
5. Use center x-coordinate as separator position

**Robustness**:
- Merging close separators (future enhancement)
- Threshold-based filtering avoids noise

### Band Detection (MVP)

**Strategy**: Mask-only (no grayscale)

**Steps**:
1. Crop lane mask: remove top/bottom %, left/right margins
2. Compute row-score: `mean(lane_mask == 0)` per row (fraction of band pixels)
3. Smooth row-scores: `cv2.blur(..., (1, kernel_h))`
4. Decide:
   - If `max(row_score) > peak_thr` AND `longest_true_run(row_score > run_thr) >= min_run`:
     - Return `True` (band detected)
   - Else: `False` (white)

**Parameters** (MVP defaults):
- `top_crop_frac = 0.15`
- `bottom_crop_frac = 0.15`
- `edge_margin_px = 6`
- `smooth_kernel_h = 9`
- `peak_thr = 0.12`
- `run_thr = 0.08`
- `min_run = 10`

**Decision Logic**:
```
band_detected = (peak_ok AND run_ok)
  where:
    peak_ok = max(row_score_smooth) > peak_thr
    run_ok = longest_consecutive_true_runs(row_score_smooth > run_thr) >= min_run
```

---

## Detector Registry

### Purpose

Allow easy swapping of band detectors without changing pipeline code.

### Registration

```python
# src/neobio/blot/band_detection.py
DETECTORS = {
    "mask": has_band_mask,
}
```

### Usage

**CLI**:
```bash
PYTHONPATH=src python scripts/run_blot.py image.jpg --band-mode mask
```

**API**:
```python
result = run_blot_pipeline(image, band_mode="mask")
```

### Adding New Detector

1. Define function:
   ```python
   def my_detector(lane_mask: np.ndarray) -> bool:
       # Must return True or False
       pass
   ```

2. Register:
   ```python
   DETECTORS["my_mode"] = my_detector
   ```

3. Use immediately:
   ```bash
   PYTHONPATH=src python scripts/run_blot.py image.jpg --band-mode my_mode
   ```

---

## Design Decisions

### Why Mask-Only (MVP)?

**Pros**:
- Simpler logic (no grayscale interpretation needed)
- Fewer hyperparameters
- More interpretable (binary mask = clear black/white regions)
- Faster (no Otsu thresholding per lane)

**Cons**:
- Less flexible for edge cases
- No intensity information used

**Decision**: MVP prioritizes simplicity and robustness; can add grayscale detector later.

### Why Binary ROI Instead of Multi-Stage?

**Decision**: Simple pipeline chosen because:
- Task is well-constrained (lanes are vertical)
- Binary mask + morphology is fast and interpretable
- Scales to large batches

### Why Detector Registry Pattern?

**Rationale**:
- Easy to add new detectors without modifying pipeline
- CLI users can select modes at runtime
- Research-friendly (compare detectors)

### Parameter Defaults

**Philosophy**: Make defaults work for typical Western blots, but allow tuning.

**Strategy**:
- Conservative defaults (lower thresholds)
- Document each parameter
- Easy override in API/CLI

---

## Testing & Validation

### Unit Tests (Future)

```python
# tests/test_lane_mask.py
def test_auto_grey_range():
    ...

# tests/test_band_detection.py
def test_has_band_mask_with_band():
    ...

def test_has_band_mask_without_band():
    ...
```

### Integration Tests

```bash
# Batch process and check output format
PYTHONPATH=src python scripts/test_blot_batch.py --input-dir test_images --output-dir test_output
```

### Manual Validation

1. Visual inspection of debug images
2. Compare with legacy detector output
3. Sensitivity/specificity analysis on labeled dataset

---

## Performance Characteristics

### Computational Complexity

| Stage | Complexity | Notes |
|-------|-----------|-------|
| Lane Mask | O(H×W) | Morphology ops are constant-time kernels |
| Separator Detection | O(H×W) | Dominated by contour finding |
| Band Detection | O(N×L×W) | N lanes, each L×W crop |
| **Total Pipeline** | **O(H×W + N×L×W)** | Typically N≤12, so ~O(H×W) |

### Bottlenecks (Profiling-Ready)

Expected hot paths for optimization:
1. `cv2.morphologyEx` (multiple passes) — 40-60% of runtime
2. `cv2.findContours` — 15-25% of runtime
3. Per-lane `cv2.blur` in band detection — 10-20% of runtime

**Optimization Strategy**: If batch processing is slow, parallelize per-image processing (trivially parallelizable).

### Memory Usage

- Peak memory: ~3× input image size (original + mask + debug overlay)
- Typical: 1920×1080 image → ~20 MB RAM
- Batch mode: Can process images serially to cap memory

### Scalability Notes

- **Single image**: Sub-second on modern hardware
- **Batch (100 images)**: Linear scaling; ~30-60 seconds on CPU
- **Future**: GPU acceleration for morphology ops could provide 5-10× speedup

---

## References

- OpenCV documentation: [cv2.morphologyEx](https://docs.opencv.org/master/d9/df8/tutorial_erosion_dilation.html)
- Morphological operations: [SciPy ndimage](https://docs.scipy.org/doc/scipy/reference/ndimage.html)
- Western blot image analysis: [ImageJ plugins](https://imagej.net/ij/plugins/)
