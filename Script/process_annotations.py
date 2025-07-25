import os, json, requests, cv2
from ultralytics import YOLO
import numpy as np
from urllib.parse import urlparse
import argparse
import sys
import re

def load_api_token():
    try:
        with open("token.txt", "r") as f:
            content = f.read()
            match = re.search(r"Your legacy token:\s*([a-f0-9]{40})", content)
            if match:
                return match.group(1)
            else:
                print("❌ Error: Could not find a valid token in 'token.txt'. Expected format: 'Your legacy token: <token>'")
                sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: 'token.txt' not found. Please create it with your API token.")
        sys.exit(1)

API_TOKEN = load_api_token()

# PROJECT_ID will be set from command line arguments
BASE_URL = "https://app.humansignal.com"
YOLO_CAR = "Models\yolov8n.pt"
YOLO_PLATE = "Models\license_plate_detector.pt"

HEADERS = {"Authorization": f"Token {API_TOKEN}"}

def fetch_tasks():
    """Export tasks from Label Studio with pagination support"""
    print("📥 Fetching tasks from Label Studio...")
    all_tasks = []
    page = 1
    page_size = 100  
    
    while True:
        try:
            print(f"  Fetching page {page}...")
            resp = requests.get(
                f"{BASE_URL}/api/projects/{PROJECT_ID}/tasks?page={page}&page_size={page_size}&fields=id,data,total_annotations,cancelled_annotations&resolve_uri=true",
                headers=HEADERS,
                timeout=120
            )
            
            if resp.status_code == 404:
                print(f"  Reached end of pagination at page {page}")
                break
                
            resp.raise_for_status()
            tasks = resp.json()
            
            all_tasks.extend(tasks)
            
            if len(tasks) < page_size:
                break
                
            page += 1
            
        except requests.exceptions.HTTPError as e:
            print(f"❌ Error: {e}")
            if all_tasks:
                print(f"  Returning {len(all_tasks)} tasks collected before the error")
                break
            else:
                raise  # Re-raise if we haven't collected any tasks
    
    print(f"  Total tasks fetched: {len(all_tasks)}")
    return all_tasks

def check_existing_annotations(task):
    """Check if a task already has annotations or was skipped"""
    try:
        task_id = task["id"]

        # Check task-level summary fields
        if task.get("cancelled_annotations", 0) > 0:
            print(f"⏭️ Task {task_id} was skipped by user")
            return True

        if task.get("total_annotations", 0) > 0:
            print(f"⏭️ Task {task_id} already has annotations")
            return True

        return False
    except Exception as e:
        print(f"⚠️ Could not check annotation status for task {task.get('id', 'unknown')}: {e}")
        return False

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
    
    if label == "License plate":  # Note the lowercase 'p' here
        base_annotation["value"].update({
            "plate_type": "Standard",  # Default value
            "plate_number": ""         # Empty as it requires OCR
        })
    
    return base_annotation

def fetch_image(url, task_id=None):
    
    # Handle resolve URLs from Label Studio when using resolve_uri=true
    if '/resolve/' in url or ('/tasks/' in url and 'fileuri=' in url):
        try:
            # If it's a relative URL, make it absolute
            if not url.startswith('http'):
                full_url = BASE_URL + url
            else:
                full_url = url
                
            print(f"🔄 Fetching from resolve URL: {full_url}")
            resp = requests.get(full_url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            print(f"❌ Failed to fetch from resolve URL: {url} — {e}")
            return None
    # Handle direct HTTP/HTTPS URLs
    elif url.startswith('http://') or url.startswith('https://') and 's3.amazonaws.com' not in url:
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            print(f"❌ Failed to fetch from direct URL: {url} — {e}")
            return None
    # Handle S3 URLs - try to convert to a resolve URL first if we have a task_id
    elif url.startswith('s3://') or 's3.amazonaws.com' in url:
        if task_id:
            # Construct a resolve URL using the task ID
            resolve_url = f"{BASE_URL}/tasks/{task_id}/resolve/?fileuri={url}"
            try:
                resp = requests.get(resolve_url, headers=HEADERS, timeout=60)
                resp.raise_for_status()
                return resp.content
            except Exception as e:
                print(f"❌ Failed to fetch from converted resolve URL: {resolve_url} — {e}")
                # Fall back to direct S3 access
                return None
        else:
            # Fall back to direct S3 access
            return None

    # Handle relative URLs from Label Studio
    else:
        return fetch_from_label_studio(url)

def fetch_from_label_studio(url):
    full_url = BASE_URL + "/data/" + url
    try:
        resp = requests.get(full_url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"❌ Failed to fetch from Label Studio: {full_url} — {e}")
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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process annotations for Label Studio tasks')
    parser.add_argument('--project-id', type=int, required=True, help='Label Studio project ID')
    return parser.parse_args()

def validate_project_id(project_id):
    """Validate that the project ID exists in Label Studio"""
    try:
        print(f"🔍 Validating project ID: {project_id}...")
        resp = requests.get(
            f"{BASE_URL}/api/projects/{project_id}",
            headers=HEADERS,
            timeout=30
        )
        
        if resp.status_code == 404:
            print(f"❌ Error: Project ID {project_id} not found in Label Studio")
            print("Please check that the project ID is correct and that you have access to it.")
            return False
            
        resp.raise_for_status()
        project_info = resp.json()
        print(f"✅ Project validated: {project_info.get('title', 'Unknown project')}")
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ Error validating project: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error validating project: {e}")
        return False

def main():
    # Parse command line arguments
    args = parse_arguments()
    global PROJECT_ID
    PROJECT_ID = args.project_id
    
    print(f"🚀 Starting annotation process for project ID: {PROJECT_ID}")
    
    # Validate project ID before proceeding
    if not validate_project_id(PROJECT_ID):
        print("❌ Aborting due to invalid project ID")
        sys.exit(1)
    
    # Load models
    print("🔄 Loading models...")
    car_model = YOLO(YOLO_CAR)
    plate_model = YOLO(YOLO_PLATE)
    

    
    # Fetch tasks
    try:
        tasks = fetch_tasks()
        if not tasks:
            print("⚠️ No tasks found in the project. Please check if the project contains any tasks.")
            return
        print(f"📋 Found {len(tasks)} tasks")
    except Exception as e:
        print(f"❌ Error fetching tasks: {e}")
        sys.exit(1)

    # Track statistics
    skipped_tasks = 0
    processed_tasks = 0

    # Process each task
    for task in tasks:
        try:
            task_id = task["id"]
            img_url = task["data"]["image"]
            
            # Check if task already has annotations
            if check_existing_annotations(task):
                print(f"⏭️ Skipping task {task_id} - already has annotations")
                skipped_tasks += 1
                continue
                
            print(f"▶ Processing task {task_id}")
            processed_tasks += 1

            # Fetch and process image
            img_bytes = fetch_image(img_url, task_id)
            if not img_bytes:
                print(f"⚠️ Could not fetch image for task {task_id}")
                continue

            img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img_array is None:
                print(f"⚠️ Could not decode image {img_url} for task {task_id}")
                continue

            # Detect objects
            results = detect(img_array, car_model, plate_model)
            if results:
                # Post annotations directly
                annotation = {"result": results}
                post_annotation(task_id, annotation)
            else:
                print(f"ℹ️ No objects detected in task {task_id}")
        except KeyError as e:
            print(f"⚠️ Missing data in task {task.get('id', 'unknown')}: {e}")
            continue
        except Exception as e:
            print(f"⚠️ Error processing task {task.get('id', 'unknown')}: {e}")
            continue

    print(f"\n✅ Processing complete! Processed {processed_tasks} tasks, skipped {skipped_tasks} tasks with existing annotations.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)