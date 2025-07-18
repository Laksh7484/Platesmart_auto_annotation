import os, csv, json, requests, cv2
from ultralytics import YOLO
import easyocr
import numpy as np
from io import BytesIO
from PIL import Image

# ─── CONFIG ─────────────────────────────────────────────────────
API_TOKEN = "5ab72f2af64a3e5dbb2d7fe64cc9ae8bc057207e"
PROJECT_ID = 168291
BASE_URL = "https://app.humansignal.com"
YOLO_CAR = "yolov8n.pt"
YOLO_PLATE = "license_plate_detector.pt"
# ────────────────────────────────────────────────────────────────

HEADERS = {"Authorization": f"Token {API_TOKEN}"}

def fetch_tasks():
    """Export tasks from Label Studio"""
    print("📥 Fetching tasks from Label Studio...")
    resp = requests.get(
        f"{BASE_URL}/api/projects/{PROJECT_ID}/tasks?page_size=10000&fields=id,data",
        headers=HEADERS,
        timeout=120
    )
    resp.raise_for_status()
    return resp.json()

def normalize(x1, y1, x2, y2, w, h):
    xc, yc = (x1+x2)/2/w, (y1+y2)/2/h
    bw, bh = (x2-x1)/w, (y2-y1)/h
    return xc, yc, bw, bh

def to_rect(xc, yc, bw, bh, label, score):
    # Convert "License Plate" to "License plate" to match the configuration
    if label == "License Plate":
        label = "License plate"

    base_annotation = {
        "from_name": "labels",  # Changed from "label" to "labels"
        "to_name": "image",
        "type": "rectanglelabels",
        "score": float(score),
        "value": {
            "x": float(round((xc - bw / 2) * 100, 4)),
            "y": float(round((yc - bh / 2) * 100, 4)),
            "width": float(round(bw * 100, 4)),
            "height": float(round(bh * 100, 4)),
            "rectanglelabels": [label]
        }
    }
    
    # Add default values for car attributes
    if label == "Car":
        base_annotation["value"].update({
            "car_make": "Toyota",  # Default value
            "car_type": "Sedan",   # Default value
            "car_color": "White"   # Default value
        })
    
    # Add default values for license plate attributes
    if label == "License plate":  # Note the lowercase 'p' here
        base_annotation["value"].update({
            "plate_type": "Standard",  # Default value
            "plate_number": ""         # Empty as it requires OCR
        })
    
    return base_annotation

def fetch_image(url):
    full_url = BASE_URL + "/data/" + url
    try:
        resp = requests.get(full_url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"❌ Failed to fetch {full_url} — {e}")
        return None

def detect(img_array, car_model, plate_model):
    h, w, _ = img_array.shape
    results = []

    # Detect car
    boxes_car = car_model(img_array)[0].boxes.cpu()
    if len(boxes_car):
        best = boxes_car.data.numpy()[boxes_car.conf.argmax()]
        x1, y1, x2, y2, conf, _ = best
        results.append(to_rect(*normalize(x1, y1, x2, y2, w, h), "Car", conf))

    # Detect plate
    boxes_plate = plate_model(img_array)[0].boxes.cpu()
    if len(boxes_plate):
        best = boxes_plate.data.numpy()[boxes_plate.conf.argmax()]
        x1, y1, x2, y2, conf, _ = best
        results.append(to_rect(*normalize(x1, y1, x2, y2, w, h), "License Plate", conf))

    return results

def post_annotation(task_id, annotation):
    url = f"{BASE_URL}/api/tasks/{task_id}/annotations"
    resp = requests.post(url, headers=HEADERS, json=annotation)
    if resp.ok:
        print(f"✅ Uploaded to task {task_id}")
    else:
        print(f"❌ Error for task {task_id}: {resp.status_code} — {resp.text}")

def main():
    # Load models
    print("🔄 Loading models...")
    car_model = YOLO(YOLO_CAR)
    plate_model = YOLO(YOLO_PLATE)
    
    # Fetch tasks
    tasks = fetch_tasks()
    print(f"📋 Found {len(tasks)} tasks")

    # Process each task
    for task in tasks:
        task_id = task["id"]
        img_url = task["data"]["image"]
        print(f"▶ Processing task {task_id}")

        # Fetch and process image
        img_bytes = fetch_image(img_url)
        if not img_bytes:
            continue

        img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            print(f"⚠️ Could not decode image {img_url}")
            continue

        # Detect objects
        results = detect(img_array, car_model, plate_model)
        if results:
            # Post annotations directly
            annotation = {"result": results}
            post_annotation(task_id, annotation)

    print("\n✅ All tasks processed!")

if __name__ == "__main__":
    main()