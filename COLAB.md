# Colab Training Workflow

This project can be trained on Google Colab with minimal changes.

## What To Upload To Google Drive

Create a Drive folder, for example:

```text
MyDrive/CV_project_colab/
```

Put these files in it:

- `motorcycle-rider.yolov8-obb.zip`
- `Helmet Wearing Image Dataset.zip`
- `two-wheeler.yolov8-obb.zip`
- `Two Wheeler Number - License Pla.yolov8-obb.zip`

Also put the project code there, either by:

1. uploading this repo folder, or
2. pushing it to GitHub and cloning it in Colab

If uploading manually, make sure these files are present in the Colab workspace:

- `dataset_tools/`
- `scripts/`
- `solution.py`
- `requirements.txt`
- `README.md`

## Notebook Cells

### 1. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Move Into Project Directory

If you uploaded the repo folder to Drive:

```bash
%cd /content
!cp -r "/content/drive/MyDrive/CV_project_colab" /content/cv_project
%cd /content/cv_project
```

If you cloned from GitHub:

```bash
%cd /content
!git clone <YOUR_REPO_URL> cv_project
%cd /content/cv_project
!cp "/content/drive/MyDrive/CV_project_colab/motorcycle-rider.yolov8-obb.zip" .
!cp "/content/drive/MyDrive/CV_project_colab/Helmet Wearing Image Dataset.zip" .
!cp "/content/drive/MyDrive/CV_project_colab/two-wheeler.yolov8-obb.zip" .
!cp "/content/drive/MyDrive/CV_project_colab/Two Wheeler Number - License Pla.yolov8-obb.zip" .
```

### 3. Install Dependencies

```bash
!pip install -r requirements.txt
```

Optional, if Colab complains about a package version mismatch:

```bash
!pip install ultralytics easyocr pillow
```

### 4. Prepare Existing YOLO Datasets

```bash
!mkdir -p data/tmp/license_plate data/tmp/traffic_violation
!unzip -qo "Two Wheeler Number - License Pla.yolov8-obb.zip" -d data/tmp/license_plate
!unzip -qo "two-wheeler.yolov8-obb.zip" -d data/tmp/traffic_violation
!python data/convert_merge.py
!python data/split_val.py
```

### 5. Prepare Downloaded Datasets

```bash
!python dataset_tools/prepare_datasets.py
```

### 6. Check GPU

```bash
!nvidia-smi
```

### 7. Train Rider Detector

This is the first model you should train.

```bash
!python scripts/train_rider_detector.py --model yolov8n.pt --imgsz 640 --batch 16 --epochs 50 --device 0
```

If Colab gives you a smaller GPU or you get OOM:

```bash
!python scripts/train_rider_detector.py --model yolov8n.pt --imgsz 512 --batch 8 --epochs 50 --device 0
```

### 8. Train Violation Detector

```bash
!python scripts/train_violation_detector.py --model yolov8n.pt --imgsz 640 --batch 16 --epochs 50 --device 0
```

Fallback:

```bash
!python scripts/train_violation_detector.py --model yolov8n.pt --imgsz 512 --batch 8 --epochs 50 --device 0
```

### 9. Prefetch OCR Weights

```bash
!mkdir -p models
!python scripts/prefetch_easyocr.py
```

### 10. Save Best Weights

```bash
!mkdir -p models
!cp runs/rider_detector/yolov8n/weights/best.pt models/rider_detector.pt
!cp runs/violation_detector/yolov8n/weights/best.pt models/violation_detector.pt
```

Copy them back to Drive:

```bash
!cp -r models "/content/drive/MyDrive/CV_project_colab/"
```

## Quick Test In Colab

```python
from solution import TrafficViolationDetector

detector = TrafficViolationDetector("./models")
result = detector.predict("path/to/test_image.jpg")
print(result)
```

## Recommended Order

1. Train rider detector
2. Evaluate a few images manually
3. Train violation detector
4. Download OCR weights
5. Copy `models/` back to your local machine

## Bring Models Back To Local Machine

When training is done, your local project should contain:

```text
models/rider_detector.pt
models/violation_detector.pt
models/easyocr/
```

Then local inference will work with:

```bash
./.venv/bin/python - <<'PY'
from solution import TrafficViolationDetector
detector = TrafficViolationDetector("./models")
print("ready")
PY
```
