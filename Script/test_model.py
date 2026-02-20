import os
import cv2
from ultralytics import YOLO

"""
Test Custom YOLO Model

This script tests your trained custom model on a few sample images
to verify it's detecting cars and license plates correctly.
"""

MODEL_PATH = "Models/custom_car_plate.pt"
TEST_IMAGES_DIR = "export_227629_project-227629-at-2026-02-12-10-40-f229e6cc/images"
OUTPUT_DIR = "test_results"
NUM_TEST_IMAGES = 10  # Number of images to test

def test_model():
    """Test the custom model on sample images"""
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Please train the model first using 'python Script/train_yolo.py'")
        return
    
    # Load model
    print(f"📦 Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get test images
    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"❌ No images found in: {TEST_IMAGES_DIR}")
        return
    
    # Test on first N images
    test_files = image_files[:NUM_TEST_IMAGES]
    
    print(f"\n🔍 Testing on {len(test_files)} images...\n")
    
    for idx, img_file in enumerate(test_files, 1):
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        
        # Run inference
        results = model(img_path, conf=0.25, verbose=False)
        
        # Get detection counts
        detections = results[0].boxes
        num_detections = len(detections)
        
        # Count by class
        car_count = sum(1 for box in detections if int(box.cls[0]) == 0)
        plate_count = sum(1 for box in detections if int(box.cls[0]) == 1)
        
        print(f"[{idx}/{len(test_files)}] {img_file}")
        print(f"   Cars: {car_count}, Plates: {plate_count}")
        
        # Save annotated image
        annotated = results[0].plot()
        output_path = os.path.join(OUTPUT_DIR, f"result_{idx}_{img_file}")
        cv2.imwrite(output_path, annotated)
    
    print(f"\n✅ Test complete! Results saved to: {OUTPUT_DIR}")
    print(f"   Review the annotated images to verify model performance.")

if __name__ == "__main__":
    test_model()
