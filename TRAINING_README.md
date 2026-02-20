# Training Custom YOLO Model for Car and License Plate Detection

This guide walks you through training a custom YOLOv8 model using your 2000+ manually annotated images from Label Studio.

## 📋 Prerequisites

- **GPU Recommended**: Training will be MUCH faster with an NVIDIA GPU with CUDA support
- **Python packages**: All required packages should already be installed (ultralytics, opencv-python, etc.)
- **Label Studio Export**: Your export folder `export_227629_project-227629-at-2026-02-12-10-40-f229e6cc` with images and labels

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Dataset
This script matches your labels (which have UUID prefixes) to images and organizes them into YOLO format:

```bash
python Script/prepare_dataset.py
```

**What it does:**
- Matches labels like `02263b54__[JZ-NRPWPA10044][...].txt` to images like `[JZ-NRPWPA10044][...].jpg`
- Remaps class IDs (Label Studio exports class 14/15, YOLO needs 0/1)
- Splits data into 80% train / 20% validation
- Creates `dataset/` folder with proper YOLO structure
- Generates `dataset/data.yaml` configuration file

### Step 2: Train Model
This will train the YOLOv8 model on your annotated data:

```bash
python Script/train_yolo.py
```

**Training parameters (can be edited in `train_yolo.py`):**
- **Epochs**: 100 (increase to 200-300 for better results)
- **Batch size**: 16 (reduce to 8 or 4 if you run out of GPU memory)
- **Image size**: 640
- **Base model**: YOLOv8n (nano, fastest - good starting point)

**Expected training time:**
- With GPU: ~30-60 minutes for 100 epochs
- With CPU: Several hours (not recommended)

**Output:**
- Trained model saved to: `Models/custom_car_plate.pt`
- Training metrics and plots: `runs/train/car_plate_detector/`

### Step 3: Test Model
Verify the model works correctly:

```bash
python Script/test_model.py
```

This will run inference on 10 sample images and save annotated results to `test_results/` folder.

## 📊 Understanding Training Results

After training, check `runs/train/car_plate_detector/` for:
- **`results.png`**: Training/validation loss curves
- **`confusion_matrix.png`**: Classification performance
- **`PR_curve.png`**: Precision-Recall curve
- **`F1_curve.png`**: F1 score vs confidence threshold

**Key metrics to look for:**
- **mAP50**: Should be > 0.80 for good performance
- **mAP50-95**: Overall accuracy metric
- **Loss curves**: Should decrease and plateau (not still dropping sharply)

## 🔧 Troubleshooting

### Class ID Mismatch
If you see warnings about unknown class IDs during preparation, check what class IDs Label Studio actually exported:

```bash
# Look at a sample label file
type export_227629_project-227629-at-2026-02-12-10-40-f229e6cc\labels\<any-label-file>.txt
```

The first number is the class ID. Update `CLASS_MAPPING` in `prepare_dataset.py` if needed.

### Out of Memory during Training
Reduce batch size in `train_yolo.py`:
- Try `BATCH_SIZE = 8`
- Or `BATCH_SIZE = 4`
- Or `BATCH_SIZE = 2` (slowest but works on any GPU)

### Poor Performance
- **More epochs**: Increase to 200-300
- **Larger model**: Change `MODEL_SIZE = "yolov8s.pt"` or `"yolov8m.pt"`
- **Data issues**: Review training images - are annotations correct?

## 🎯 After Training: Update process_annotations.py

Once you're happy with the model, update `process_annotations.py` to use it:

1. Replace the separate car and plate models with your custom model:

```python
# OLD (around line 108-109)
CAR_MODEL = YOLO("Models/yolov8n.pt")
PLATE_MODEL = YOLO("Models/license_plate_detector.pt")

# NEW
CUSTOM_MODEL = YOLO("Models/custom_car_plate.pt")
```

2. Update the `detect_with_yolo` function to use the new model (both car and plate in one pass)

## 📁 File Structure

```
Platesmart_auto_annotation/
├── Script/
│   ├── prepare_dataset.py    # Step 1: Organize data
│   ├── train_yolo.py          # Step 2: Train model
│   └── test_model.py          # Step 3: Test model
├── Models/
│   ├── yolov8n.pt             # Base model (pretrained)
│   └── custom_car_plate.pt    # Your trained model (created after training)
├── dataset/                   # Created by prepare_dataset.py
│   ├── images/
│   │   ├── train/            # 80% of images
│   │   └── val/              # 20% of images
│   ├── labels/
│   │   ├── train/            # 80% of labels
│   │   └── val/              # 20% of labels
│   └── data.yaml             # YOLO config file
├── runs/                      # Created by train_yolo.py
│   └── train/
│       └── car_plate_detector/
│           ├── weights/
│           │   └── best.pt   # Best model checkpoint
│           └── *.png         # Training plots
└── test_results/              # Created by test_model.py
    └── result_*.jpg          # Annotated test images
```

## ❓ Questions?

- Check if GPU is being used: `nvidia-smi` (Windows) or shows GPU info when training starts
- Monitor GPU usage: Task Manager > Performance > GPU
- Review training progress: Training will show loss/mAP metrics every epoch

---

**Author**: Auto-generated training pipeline  
**Date**: 2026-02-12
