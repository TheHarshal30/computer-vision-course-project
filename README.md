# Traffic Violation Detector

This workspace now has a minimal end-to-end pipeline for the course project:

- `datasets/processed/helmet_wearing`: classification dataset derived from the Mendeley helmet archive
- `data/merged_dataset`: existing YOLO detection dataset for `helmet`, `no-helmet`, `motorcycle`, `license_plate`
- `solution.py`: required `TrafficViolationDetector` class

## Dataset Preparation

Run this after downloading or replacing the raw archives:

```bash
./.venv/bin/python dataset_tools/prepare_datasets.py
```

## Training

Train the violation detector:

```bash
./.venv/bin/python scripts/train_violation_detector.py
```

Optional helmet classifier training on the Mendeley classification data:

```bash
./.venv/bin/python scripts/train_helmet_classifier.py
```

Prefetch OCR weights into the local `models/` folder so inference stays offline-compatible:

```bash
./.venv/bin/python scripts/prefetch_easyocr.py
```

## Model Placement

Place trained weights in `./models` with these names:

- `models/violation_detector.pt`
- `models/person_detector.pt`

`solution.py` looks for those exact filenames first.

For `models/person_detector.pt`, you can reuse a COCO-pretrained YOLO checkpoint, for example:

```bash
cp yolov8n.pt models/person_detector.pt
```

## Current Inference Logic

`solution.py` uses:

1. COCO `person` detector for rider counting
2. violation detector for `motorcycle`, `no-helmet`, and `license_plate`
3. OCR via `easyocr` loaded from `models/easyocr`, using GPU automatically when CUDA is available

If OCR weights are missing, `license_plate` falls back to an empty string instead of failing.
