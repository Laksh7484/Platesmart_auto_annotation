import os
from ultralytics import YOLO
import torch

"""
YOLO Training Script for Car and License Plate Detection

This script trains a custom YOLOv8 model on your annotated dataset.
The trained model will detect both Cars and License Plates.

Requirements:
- Run prepare_dataset.py first to organize your data
- GPU recommended (training on CPU is extremely slow)
- dataset/data.yaml must exist
"""

# Configuration
DATA_YAML = "dataset/data.yaml"
MODEL_SIZE = "yolov8n.pt"  # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
OUTPUT_MODEL = "Models/custom_car_plate.pt"
EPOCHS = 100  # Increase for better results (100-300)
IMAGE_SIZE = 640  # Image size for training
BATCH_SIZE = 16  # Reduce if you run out of GPU memory (try 8, 4, or even 2)
DEVICE = 0  # 0 for GPU, 'cpu' for CPU

def check_prerequisites():
    """Check if dataset is prepared"""
    if not os.path.exists(DATA_YAML):
        print(f"❌ Error: {DATA_YAML} not found!")
        print("   Please run 'python Script/prepare_dataset.py' first to prepare the dataset.")
        return False
    
    if not os.path.exists("dataset/images/train"):
        print("❌ Error: Training images not found!")
        print("   Please run 'python Script/prepare_dataset.py' first.")
        return False
    
    return True

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {gpu_name}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠️  No GPU detected. Training will run on CPU (very slow).")
        print("   Consider using Google Colab or a machine with GPU for faster training.")
        return False

def train_model():
    """Train the YOLO model"""
    print("\n" + "=" * 60)
    print("Starting YOLO Training")
    print("=" * 60)
    
    # Load pretrained model
    print(f"\n📦 Loading base model: {MODEL_SIZE}")
    model = YOLO(f"Models/{MODEL_SIZE}")
    
    # Training parameters
    print(f"\n⚙️  Training Configuration:")
    print(f"   Dataset: {DATA_YAML}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Image Size: {IMAGE_SIZE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Device: {'GPU' if DEVICE == 0 else 'CPU'}")
    
    # Start training
    print(f"\n🚀 Starting training... (this may take a while)")
    print("=" * 60)
    
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project="runs/train",
        name="car_plate_detector",
        exist_ok=True,
        patience=20,  # Early stopping patience
        save=True,
        plots=True,  # Save training plots
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("✅ Training completed!")
    print("=" * 60)
    
    return results

def save_best_model():
    """Copy the best model to the Models directory"""
    # YOLOv8 saves the best model as 'best.pt' in the run directory
    best_model_path = "runs/train/car_plate_detector/weights/best.pt"
    
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy2(best_model_path, OUTPUT_MODEL)
        print(f"\n✅ Best model saved to: {OUTPUT_MODEL}")
        print(f"   Original location: {best_model_path}")
    else:
        print(f"\n⚠️  Best model not found at: {best_model_path}")
        print("   Please check the training output directory.")

def validate_model():
    """Run validation on the trained model"""
    print("\n" + "=" * 60)
    print("Running Validation")
    print("=" * 60)
    
    model = YOLO(OUTPUT_MODEL)
    results = model.val(data=DATA_YAML, split='val')
    
    print(f"\n📊 Validation Results:")
    print(f"   mAP50: {results.box.map50:.4f}")
    print(f"   mAP50-95: {results.box.map:.4f}")
    
    return results

def main():
    print("=" * 60)
    print("YOLO Model Training Script")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Check GPU
    has_gpu = check_gpu()
    if not has_gpu:
        response = input("\nContinue training on CPU? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
        global DEVICE
        DEVICE = 'cpu'
    
    # Train model
    try:
        train_model()
        save_best_model()
        
        # Validate
        if os.path.exists(OUTPUT_MODEL):
            validate_model()
            
            print("\n" + "=" * 60)
            print("🎉 Training Complete!")
            print("=" * 60)
            print(f"\n📁 Model saved: {OUTPUT_MODEL}")
            print(f"📁 Training results: runs/train/car_plate_detector/")
            print("\nNext steps:")
            print("  1. Review training metrics in 'runs/train/car_plate_detector/'")
            print("  2. Test the model with 'python Script/test_model.py'")
            print("  3. Update 'Script/process_annotations.py' to use the new model")
            print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
