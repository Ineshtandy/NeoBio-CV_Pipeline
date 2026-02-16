# NeoBio-CV_Pipeline

Computer-vision pipeline for western blot lane segmentation and per-lane band presence detection.

## Current capabilities

- Build a binary lane mask from a white-background blot image.
- Detect the blot ROI and vertical separators to generate lane boxes.
- For each lane, determine whether a dark band is present.
- Output a debug image with ROI and per-lane BAND/WHITE labels.

## Core script

All logic currently lives in [blot_detect.py](blot_detect.py).

### Lane mask creation

`build_lane_mask(image)` generates a binary mask:

- white (255): lane/blot region
- black (0): separators and background

Mask creation steps:

1. Convert to grayscale.
2. Compute adaptive grey range via `auto_grey_range()`.
3. `inRange` to isolate lane grey region.
4. Morphology (open to remove specks, then invert + vertical close + dilate + invert) to strengthen separator lines.

### Lane detection

Lane boundaries are obtained by:

1. Computing a ROI from the mask (`find_roi_from_mask`).
2. Detecting vertical separator lines on the ROI (`find_vertical_separators`).
3. Converting separators to lane boxes (`lane_ranges_from_separators`).

### Band detection

Band presence is computed per lane. Two modes are supported:

- `grayscale` (default): Otsu threshold on grayscale lane crop, then mask to blot region.
- `mask`: mask-only mode that detects dark rows inside the binary mask.

The mode is controlled via `--band-mode`.

## Usage

Single image:

```
python blot_detect.py "NeoBio Input Images/your_image.jpeg"
```

Mask-only band detection:

```
python blot_detect.py "NeoBio Input Images/your_image.jpeg" --band-mode mask
```

Batch test run over all images in `NeoBio Input Images/`:

```
python blot_detect.py
```

## Outputs

Debug images are written to:

- `./testing output images/<filename>_debug_blot.png`

These include:

- ROI bounding box
- Lane boxes
- Per-lane labels (BAND/WHITE)

## Notes

- Temporary debug artifacts like `mask_post_morph.png` and `mask_prior_morph.png` are ignored by git.
- The input image folder `NeoBio Input Images/` is ignored by git.

## Morphology reminders

- **Open**: erosion then dilation (removes small white noise)
- **Close**: dilation then erosion (fills small gaps)