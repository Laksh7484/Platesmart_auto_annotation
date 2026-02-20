import os, json, requests, cv2
import numpy as np
import argparse
import sys
import re
import base64
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from ultralytics import YOLO

# Configuration
BASE_URL = "https://app.humansignal.com"
MODEL_PATH = "Models/custom_car_plate.pt"

# Load API token
def load_api_token():
    try:
        with open("token.txt", "r") as f:
            content = f.read()
            match = re.search(r"Your legacy token:\s*([a-f0-9]{40})", content)
            if match:
                return match.group(1)
            else:
                print("Error: Could not find a valid token in 'token.txt'. Expected format: 'Your legacy token: <token>'")
                sys.exit(1)
    except FileNotFoundError:
        print("Error: 'token.txt' not found. Please create it with your API token.")
        sys.exit(1)

API_TOKEN = load_api_token()
HEADERS = {"Authorization": f"Token {API_TOKEN}"}

# Load the custom YOLO model
print(f"Loading custom YOLO model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Custom model not found at {MODEL_PATH}. Please make sure you have trained and moved it there.")
    sys.exit(1)
CUSTOM_MODEL = YOLO(MODEL_PATH)
print("Custom YOLO model loaded successfully!")

def fetch_tasks(project_id):
    """Export tasks from Label Studio with pagination support"""
    print(f"Fetching tasks from Label Studio project {project_id}...")
    all_tasks = []
    page = 1
    page_size = 100  
    
    while True:
        try:
            print(f"  Fetching page {page}...")
            resp = requests.get(
                f"{BASE_URL}/api/projects/{project_id}/tasks?page={page}&page_size={page_size}&fields=id,data,total_annotations,cancelled_annotations&resolve_uri=true",
                headers=HEADERS,
                timeout=120
            )
            
            if resp.status_code == 404:
                break
                
            resp.raise_for_status()
            tasks = resp.json()
            
            all_tasks.extend(tasks)
            
            if len(tasks) < page_size:
                break
                
            page += 1
            
        except Exception as e:
            print(f"Error fetching tasks: {e}")
            if all_tasks:
                break
            else:
                raise
    
    # Deduplicate tasks by ID
    unique_tasks = []
    seen_ids = set()
    for t in all_tasks:
        if t['id'] not in seen_ids:
            unique_tasks.append(t)
            seen_ids.add(t['id'])
            
    print(f"  Total unique tasks to process: {len(unique_tasks)}")
    return unique_tasks

def check_existing_annotations(task):
    """Check if a task already has annotations or was skipped"""
    if task.get("cancelled_annotations", 0) > 0:
        return True
    if task.get("total_annotations", 0) > 0:
        return True
    return False

def normalize(x1, y1, x2, y2, w, h):
    xc, yc = (x1+x2)/2/w, (y1+y2)/2/h
    bw, bh = (x2-x1)/w, (y2-y1)/h
    return xc, yc, bw, bh

def fetch_image(url, task_id):
    try:
        # Resolve URL if it's internal to Label Studio
        if url.startswith('/'):
            full_url = BASE_URL + url
        elif '/resolve/' in url or ('/tasks/' in url and 'fileuri=' in url):
            full_url = url if url.startswith('http') else BASE_URL + url
        elif url.startswith('s3://') or 's3.amazonaws.com' in url:
            # Resolve S3 URLs via Label Studio's resolve endpoint
            if task_id:
                full_url = f"{BASE_URL}/tasks/{task_id}/resolve/?fileuri={url}"
            else:
                return None
        else:
            # For data/upload URLs
            full_url = BASE_URL + "/data/" + url if "upload" in url else url

        resp = requests.get(full_url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"Failed to fetch image for task {task_id}: {e}")
        return None

def detect_only_custom(img_bytes, img_width, img_height, task_id):
    """Detection using ONLY the custom YOLO model"""
    try:
        # Decode image
        img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            return []
        
        annotations = []
        
        # Run inference with custom model
        # Class 0: Car, Class 1: License-Plate (mapping from your data.yaml)
        results = CUSTOM_MODEL(img_array, conf=0.25, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = "Car" if cls_id == 0 else "License plate" if cls_id == 1 else "Unknown"
                
                if label == "Unknown":
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                xc, yc, bw, bh = normalize(x1, y1, x2, y2, img_width, img_height)
                
                # Create annotation
                ann = {
                    "id": str(uuid.uuid4())[:10],
                    "type": "rectanglelabels",
                    "value": {
                        "x": float(round((xc - bw / 2) * 100, 4)),
                        "y": float(round((yc - bh / 2) * 100, 4)),
                        "width": float(round(bw * 100, 4)),
                        "height": float(round(bh * 100, 4)),
                        "rotation": 0,
                        "rectanglelabels": [label]
                    },
                    "score": float(box.conf[0]),
                    "from_name": "labels",
                    "to_name": "image"
                }
                annotations.append(ann)
        
        return annotations
        
    except Exception as e:
        print(f"Error during custom YOLO detection for task {task_id}: {e}")
        return []

def post_annotation(task_id, results):
    url = f"{BASE_URL}/api/tasks/{task_id}/annotations"
    resp = requests.post(url, headers=HEADERS, json={"result": results})
    if resp.ok:
        print(f"✅ Uploaded to task {task_id}")
    else:
        print(f"❌ Error for task {task_id}: {resp.status_code} — {resp.text}")

def process_single_task(task, worker_id):
    try:
        task_id = task["id"]
        img_url = task["data"]["image"]
        
        if check_existing_annotations(task):
            return {"status": "skipped", "task_id": task_id}

        img_bytes = fetch_image(img_url, task_id)
        if not img_bytes:
            return {"status": "error", "task_id": task_id, "reason": "image_fetch_failed"}

        img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            return {"status": "error", "task_id": task_id, "reason": "image_decode_failed"}

        img_height, img_width = img_array.shape[:2]
        results = detect_only_custom(img_bytes, img_width, img_height, task_id)
        
        if results:
            post_annotation(task_id, results)
            return {"status": "success", "task_id": task_id}
        else:
            return {"status": "success", "task_id": task_id, "reason": "no_objects"}
            
    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        return {"status": "error", "task_id": task.get('id', 'unknown'), "reason": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Annotate tasks using ONLY custom YOLO model')
    parser.add_argument('--project-id', type=int, required=True, help='Label Studio project ID')
    parser.add_argument('--max-workers', type=int, default=7, help='Concurrency')
    args = parser.parse_args()
    
    tasks = fetch_tasks(args.project_id)
    if not tasks:
        print("No tasks found.")
        return

    processed = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_single_task, task, i % args.max_workers + 1) for i, task in enumerate(tasks)]
        for future in as_completed(futures):
            res = future.result()
            if res["status"] == "success": processed += 1
            elif res["status"] == "skipped": skipped += 1
            else: errors += 1
            
            print(f"Progress: {processed + skipped + errors}/{len(tasks)} (Succ: {processed}, Skip: {skipped}, Err: {errors})", end="\r")

    print(f"\nCompleted. Succ: {processed}, Skip: {skipped}, Err: {errors}")

if __name__ == "__main__":
    main()
