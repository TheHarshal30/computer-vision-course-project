# Traffic Violation Detector

End-to-end pipeline for the AID 728 course project: given an RGB image of a street scene,
return per-two-wheeler reports of rider count, helmet violations, and license plate text.

The system is **fully offline** at inference time — all model weights live under `./models`
and no network calls are made.

## Architecture

```
Image
  │
  ├─► YOLOv8 violation detector  ───► motorcycle, helmet, no_helmet, license_plate
  └─► YOLOv8n (COCO) person detector ─► person bboxes
            │
            ▼
   Hungarian assignment: each person → at most one motorcycle
            │
            ▼
   Per-motorcycle:
     • num_riders = assigned persons
     • helmet_violations = riders whose head region has no overlapping helmet bbox
     • license plate: aspect-ratio gate → OCR ensemble → Indian-plate regex validation
            │
            ▼
   {"violations": [{num_riders, helmet_violations, license_plate}, ...]}
```

Key design choices:

- **Inverted helmet logic**: the trained `helmet` class is much stronger than `no_helmet`,
  so violations are inferred by absence (no helmet bbox over the rider's head region).
- **Global Hungarian assignment** (`scipy.optimize.linear_sum_assignment`): each detected
  person is assigned to at most one motorcycle (preventing double-counting in multi-bike
  scenes); each motorcycle can hold up to 3 riders (motorcycle columns tiled 3× in the
  score matrix).
- **OCR ensemble**: each plate crop is read in four preprocessing variants (original,
  2× upscale, deskew, adaptive-threshold). Every reading is then validated against the
  Indian plate regex `^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$`; the highest-confidence
  regex-valid result wins. Empty string is returned when no variant produces a trusted
  reading — better than guessing under the edit-distance OCR metric.
- **Output filter**: the `violations` list contains only motorcycles that *violate* a rule
  (>2 riders OR ≥1 helmet violation). Non-violating two-wheelers are excluded.
- **Plate aspect-ratio gate**: plate detections outside `[1.3, 6.0]` aspect ratio are
  discarded before wasting OCR on them.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Models

`solution.py` expects three artifacts under `./models`:

| File | Source | Size |
|---|---|---|
| `models/violation_detector.pt` | Custom-trained YOLOv8n (see Training) | ~6 MB |
| `models/person_detector.pt` | Stock COCO YOLOv8n (`yolov8n.pt`) | ~6 MB |
| `models/easyocr/` | Prefetched EasyOCR weights (CRAFT + g2) | ~93 MB |

**Total ≈ 105 MB** — well under the 250 MB submission cap.

Set up the person detector and OCR weights once:

```bash
.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
cp yolov8n.pt models/person_detector.pt
.venv/bin/python scripts/prefetch_easyocr.py
```

## Training

The violation detector is fine-tuned from `yolov8n.pt` on a merged 4-class dataset
(11,479 images) assembled from five Roboflow sources covering helmets, no-helmet,
motorcycles, generic plates, and Indian-specific plates.

Build the merged dataset (downloads must already be in `data/tmp/`):

```bash
.venv/bin/python data/merge_new_datasets.py
```

Train on macOS (Apple Silicon, MPS):

```bash
caffeinate -i .venv/bin/python scripts/train_violation_detector.py \
    --data data/merged_dataset/data.yaml \
    --model yolov8n.pt \
    --epochs 10 \
    --imgsz 640 \
    --batch 16 \
    --workers 4 \
    --cache disk \
    --close-mosaic 3 \
    --device mps \
    --project runs/violation_detector \
    --name yolov8n
```

The script uses MPS-tuned defaults: cosine LR, AMP, 4 dataloader workers, disk cache.
On an M5 MacBook, 10 epochs over 11k images takes ~60 minutes.

On Linux with CUDA, replace `--device mps` with `--device 0` (or your GPU index). On CPU,
use `--device cpu` — but expect training to take many hours.

Copy the best weights into `./models`:

```bash
cp runs/detect/runs/violation_detector/yolov8n/weights/best.pt models/violation_detector.pt
```

## Inference

```python
from solution import TrafficViolationDetector

detector = TrafficViolationDetector("./models")
result = detector.predict("path/to/image.jpg")
print(result)
# {"violations": [{"num_riders": 3, "helmet_violations": 0, "license_plate": "KA01AB1234"}]}
```

If the image has no rule-violating two-wheeler, `result["violations"]` is `[]`.

## Training results (10 epochs, val split)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.927 | 0.933 | **0.970** | 0.707 |
| helmet | 0.926 | 0.948 | 0.970 | 0.721 |
| no_helmet | 0.909 | 0.895 | 0.954 | 0.576 |
| motorcycle | 0.907 | 0.931 | 0.969 | 0.711 |
| license_plate | 0.967 | 0.957 | 0.985 | 0.818 |

Inference speed: **17.3 ms per image** on M5 MPS (~58 FPS).

## Submission layout

```
<ROLL_NUMBER>/
  solution.py
  requirements.txt
  README.md
  models/
    violation_detector.pt
    person_detector.pt
    easyocr/
      craft_mlt_25k.pth
      english_g2.pth
```

## Files

- `solution.py` — required `TrafficViolationDetector` class and pipeline
- `scripts/train_violation_detector.py` — YOLOv8 fine-tuning script (MPS-tuned)
- `scripts/prefetch_easyocr.py` — downloads OCR weights for offline inference
- `data/merge_new_datasets.py` — merges five Roboflow datasets into the 4-class training set
- `requirements.txt` — pinned dependencies
