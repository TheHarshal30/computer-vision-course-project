# Current System Architecture And Results

This document describes the current state of the traffic-violation project as implemented in [solution.py](/home/harshal/CV%20project%20/solution.py:1), plus the main training results collected so far.

## Goal

Given one street image, return:

```python
{
  "violations": [
    {
      "num_riders": int,
      "helmet_violations": int,
      "license_plate": "string"
    }
  ]
}
```

Project constraints from the brief:

- total model size must stay under `250 MB`
- inference must run offline
- no large VLMs

## Current Architecture

The current system is a two-model detection pipeline with OCR:

1. `violation detector`
   - custom-trained YOLO detector
   - classes:
     - `helmet`
     - `no-helmet`
     - `motorcycle`
     - `license_plate`
2. `person detector`
   - COCO-pretrained YOLO model
   - only COCO class `0 = person` is used
3. `OCR`
   - EasyOCR
   - reads cropped plate regions

### Why This Design

Earlier experiments used a custom-trained `rider` detector. Validation metrics improved, but the model still failed badly on hard multi-rider violation images, especially grayscale, tilted, crowded scenes. Because of that, the current system dropped the trained rider model and replaced rider counting with a generic COCO `person` detector.

This keeps:

- custom model where domain adaptation helped:
  - `motorcycle`
  - `no-helmet`
  - `license_plate`
- generic pretrained model where custom model underperformed:
  - `person`

## Inference Flow

Current inference flow in [solution.py](/home/harshal/CV%20project%20/solution.py:1):

1. Load image with OpenCV.
2. Run `violation_model` on full image.
3. Run `person_model` on full image.
4. Collect:
   - motorcycles from violation detector
   - persons from COCO detector
   - no-helmet boxes from violation detector
   - license plates from violation detector
5. For each motorcycle:
   - assign nearby/overlapping persons to that motorcycle
   - `num_riders = assigned persons`
   - assign overlapping `no-helmet` detections to those riders
   - `helmet_violations = assigned no-helmet detections`
6. Keep only motorcycles where:
   - `num_riders > 2`, or
   - `helmet_violations > 0`
7. For each kept motorcycle:
   - choose best candidate plate
   - crop plate
   - preprocess crop
   - OCR text
   - normalize to uppercase alphanumeric
8. Return final JSON.

## Model Files

Current expected model files under `./models`:

- `violation_detector.pt`
- `person_detector.pt`
- `easyocr/` OCR weights directory

For `person_detector.pt`, a COCO-pretrained YOLO checkpoint can be reused directly, for example:

```bash
cp yolov8n.pt models/person_detector.pt
```

## Datasets Used

### 1. Violation Detector Dataset

`data/merged_dataset`

Built by merging:

- local two-wheeler violation dataset
- local license plate dataset

Final classes:

- `0 = helmet`
- `1 = no-helmet`
- `2 = motorcycle`
- `3 = license_plate`

Relevant scripts:

- [data/convert_merge.py](/home/harshal/CV%20project%20/data/convert_merge.py:1)
- [data/split_val.py](/home/harshal/CV%20project%20/data/split_val.py:1)

### 2. Helmet Classification Dataset

`datasets/processed/helmet_wearing`

Prepared from Mendeley helmet archive.

Classes:

- `correct_way`
- `incorrect_way`

This dataset is prepared and supported in the repo, but the current `solution.py` does not yet use a second-stage helmet classifier.

### 3. Rider Dataset

`datasets/processed/motorcycle_rider_obb`
`datasets/processed/motorcycle_rider_detect`

Prepared from Roboflow motorcycle-rider export.

Classes:

- `motorcycle`
- `rider`

This dataset was used for rider detector experiments, but the current production path no longer depends on that trained rider detector.

## Training Results

### Rider Detector Results

#### Run A

Validation summary:

- `all`: `P 0.819`, `R 0.635`, `mAP50 0.726`, `mAP50-95 0.424`
- `motorcycle`: `P 0.839`, `R 0.664`, `mAP50 0.762`, `mAP50-95 0.457`
- `rider`: `P 0.799`, `R 0.605`, `mAP50 0.689`, `mAP50-95 0.391`

#### Run B

Validation summary:

- `all`: `P 0.880`, `R 0.738`, `mAP50 0.801`, `mAP50-95 0.471`
- `motorcycle`: `P 0.889`, `R 0.772`, `mAP50 0.840`, `mAP50-95 0.506`
- `rider`: `P 0.872`, `R 0.703`, `mAP50 0.761`, `mAP50-95 0.436`

#### Run C

Validation summary:

- `all`: `P 0.880`, `R 0.755`, `mAP50 0.810`, `mAP50-95 0.490`
- `motorcycle`: `P 0.910`, `R 0.784`, `mAP50 0.845`, `mAP50-95 0.519`
- `rider`: `P 0.851`, `R 0.725`, `mAP50 0.776`, `mAP50-95 0.462`

### Rider Detector Conclusion

Metrics improved across runs, but qualitative testing exposed a more serious issue:

- hard multi-rider violation images still produced no detections
- especially grayscale, rotated, crowded scenes

Because of that, the custom rider detector was not considered reliable enough for current end-to-end counting, despite better validation metrics.

### Violation Detector Results

Validation summary:

- `all`: `P 0.815`, `R 0.834`, `mAP50 0.852`, `mAP50-95 0.601`
- `helmet`: `P 0.792`, `R 0.845`, `mAP50 0.882`, `mAP50-95 0.555`
- `no-helmet`: `P 0.556`, `R 0.600`, `mAP50 0.601`, `mAP50-95 0.323`
- `motorcycle`: `P 0.997`, `R 1.000`, `mAP50 0.995`, `mAP50-95 0.777`
- `license_plate`: `P 0.914`, `R 0.890`, `mAP50 0.931`, `mAP50-95 0.750`

### Violation Detector Conclusion

Strong classes:

- `motorcycle`
- `license_plate`
- `helmet`

Weak class:

- `no-helmet`

This means the current system is likely strongest on:

- motorcycle localization
- plate extraction
- OCR-ready plate crops

And weakest on:

- correctly finding all no-helmet riders

## Current Strengths

- model size remains comfortably below the project `250 MB` cap
- offline inference supported
- motorcycle detection is very strong
- plate detection is strong
- OCR path is integrated
- architecture is simple enough to debug

## Current Weaknesses

- `no-helmet` detection is the weakest important class
- dense multi-rider counting remains hard
- current rider counting depends on generic `person` detection plus heuristics
- dataset labels in `merged_dataset` are not always a reliable oracle for true end-to-end violations
- OCR quality depends heavily on crop quality and plate visibility

## Known Design Tradeoff

Current system prefers:

- better generalization for person detection via COCO
- domain-specific detection for motorcycle / no-helmet / plate

instead of:

- fully custom end-to-end detector for all subtasks

This is intentional. It reduces reliance on the weakest custom component found so far.

## Recommended Next Improvements

Most likely high-value next steps:

1. improve `no-helmet` quality
   - data audit
   - more no-helmet examples
   - possible second-stage helmet classifier
2. improve rider counting on crowded bikes
   - stronger person/rider association logic
   - possibly person pose or head-level reasoning
3. strengthen OCR on difficult plates
   - better crop selection
   - better plate preprocessing

## Files To Read

- [solution.py](/home/harshal/CV%20project%20/solution.py:1)
- [README.md](/home/harshal/CV%20project%20/README.md:1)
- [dataset_tools/prepare_datasets.py](/home/harshal/CV%20project%20/dataset_tools/prepare_datasets.py:1)
- [scripts/train_violation_detector.py](/home/harshal/CV%20project%20/scripts/train_violation_detector.py:1)
- [scripts/train_helmet_classifier.py](/home/harshal/CV%20project%20/scripts/train_helmet_classifier.py:1)
