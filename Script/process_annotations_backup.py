import os, json, requests, cv2
import numpy as np
from urllib.parse import urlparse
import argparse
import sys
import re
import base64
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Load mappings from JSON file
def load_mappings():
    try:
        with open("Script/mappings.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: 'Script/mappings.json' not found. Please create it with your mappings.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in 'Script/mappings.json': {e}")
        sys.exit(1)

# Load mappings at module level
MAPPINGS = load_mappings()

def get_state_name_from_code(region_code):
    """Convert Plate Recognizer region code (e.g. 'us-pa') to project label (e.g. 'US-PA')."""
    if not region_code:
        return None
    return MAPPINGS["state_codes"].get(region_code.lower())

def normalize_color(raw_color):
    if not raw_color:
        return "Unknown"
    color_l = raw_color.strip().lower()
    if color_l in MAPPINGS["color_synonyms"]:
        return MAPPINGS["color_synonyms"][color_l]
    # Capitalize first letter for consistency (e.g., 'red' -> 'Red')
    return color_l.capitalize()

def normalize_car_type(raw_type):
    if not raw_type:
        return "Unknown"
    t = raw_type.strip().lower()
    if t in MAPPINGS["car_type_synonyms"]:
        return MAPPINGS["car_type_synonyms"][t]
    return t.capitalize()

def normalize_make(raw_make):
    if not raw_make:
        return "Unknown"
    return raw_make.strip().capitalize()

def normalize_model(raw_model):
    if not raw_model:
        return "Unknown"
    return raw_model.strip().capitalize()

def normalize_orientation(raw_orientation):
    if not raw_orientation:
        return "Unknown"
    return raw_orientation.strip().capitalize()

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

def load_plate_recognizer_token():
    try:
        with open("plate_recognizer_token.txt", "r") as f:
            token = f.read().strip()
            if token:
                return token
            else:
                print("Error: Plate Recognizer token is empty in 'plate_recognizer_token.txt'")
                sys.exit(1)
    except FileNotFoundError:
        print("Error: 'plate_recognizer_token.txt' not found. Please create it with your Plate Recognizer API token.")
        sys.exit(1)

API_TOKEN = load_api_token()
PLATE_RECOGNIZER_TOKEN = load_plate_recognizer_token()

BASE_URL = "https://app.humansignal.com"
PLATE_RECOGNIZER_URL = "http://localhost:8080/v1/plate-reader/"

HEADERS = {"Authorization": f"Token {API_TOKEN}"}
PLATE_RECOGNIZER_HEADERS = {"Authorization": f"Token {PLATE_RECOGNIZER_TOKEN}"}

def fetch_tasks():
    """Export tasks from Label Studio with pagination support"""
    print("Fetching tasks from Label Studio...")
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
            print(f"Error: {e}")
            if all_tasks:
                print(f"  Returning {len(all_tasks)} tasks collected before the error")
                break
            else:
                raise  # Re-raise if we haven't collected any tasks
    
    print(f"  Total tasks fetched (raw): {len(all_tasks)}")
    
    # Deduplicate tasks by ID
    unique_tasks = []
    seen_ids = set()
    for t in all_tasks:
        if t['id'] not in seen_ids:
            unique_tasks.append(t)
            seen_ids.add(t['id'])
            
    if len(unique_tasks) < len(all_tasks):
        print(f"  Removed {len(all_tasks) - len(unique_tasks)} duplicate tasks")
        
    print(f"  Total unique tasks to process: {len(unique_tasks)}")
    return unique_tasks

def check_existing_annotations(task):
    """Check if a task already has annotations or was skipped"""
    try:
        task_id = task["id"]

        if task.get("cancelled_annotations", 0) > 0:
            print(f"Task {task_id} was skipped by user")
            return True

        if task.get("total_annotations", 0) > 0:
            print(f"Task {task_id} already has annotations")
            return True

        return False
    except Exception as e:
        print(f"Could not check annotation status for task {task.get('id', 'unknown')}: {e}")
        return False

def normalize(x1, y1, x2, y2, w, h):
    xc, yc = (x1+x2)/2/w, (y1+y2)/2/h
    bw, bh = (x2-x1)/w, (y2-y1)/h
    return xc, yc, bw, bh

def to_rect(xc, yc, bw, bh, label, score, additional_props=None):
    base_annotation = {
        "from_name": "labels",
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
    
    if additional_props:
        base_annotation["value"].update(additional_props)
    
    return base_annotation

def fetch_image(url, task_id=None):
    
    if '/resolve/' in url or ('/tasks/' in url and 'fileuri=' in url):
        try:
            if not url.startswith('http'):
                full_url = BASE_URL + url
            else:
                full_url = url
                
            print(f"Fetching from resolve URL: {full_url}")
            resp = requests.get(full_url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            print(f"Failed to fetch from resolve URL: {url} — {e}")
            return None
    elif url.startswith('http://') or url.startswith('https://') and 's3.amazonaws.com' not in url:
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            print(f"Failed to fetch from direct URL: {url} — {e}")
            return None
    elif url.startswith('s3://') or 's3.amazonaws.com' in url:
        if task_id:
            resolve_url = f"{BASE_URL}/tasks/{task_id}/resolve/?fileuri={url}"
            try:
                resp = requests.get(resolve_url, headers=HEADERS, timeout=60)
                resp.raise_for_status()
                return resp.content
            except Exception as e:
                print(f"Failed to fetch from converted resolve URL: {resolve_url} — {e}")
                return None
        else:
            return None

    else:
        return fetch_from_label_studio(url)

def fetch_from_label_studio(url):
    full_url = BASE_URL + "/data/" + url
    try:
        resp = requests.get(full_url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"Failed to fetch from Label Studio: {full_url} — {e}")
        return None

def detect_with_plate_recognizer(img_bytes, task_id):
    """Use Plate Recognizer Snapshot SDK (Docker) to detect vehicles and license plates"""
    try:
        time.sleep(0.1)

        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        payload = {
            "upload": img_base64,
            "regions": ["us"],
            "mmc": True,
            "vehicle": True,
            "camera_id": "auto_annotation"
        }

        files = {
            'upload': ('image.jpg', img_bytes, 'image/jpeg')
        }
        data = {
            'regions': 'us',
            'mmc': 'true',
            'vehicle': 'true',
            'camera_id': 'auto_annotation'
        }

        resp = requests.post(
            PLATE_RECOGNIZER_URL,
            headers=PLATE_RECOGNIZER_HEADERS,
            files=files,
            data=data,
            timeout=120
        )

        if resp.status_code != 200:
            resp = requests.post(
                PLATE_RECOGNIZER_URL,
                headers=PLATE_RECOGNIZER_HEADERS,
                json=payload,
                timeout=120
            )

        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.post(
                PLATE_RECOGNIZER_URL,
                headers=PLATE_RECOGNIZER_HEADERS,
                json=payload,
                timeout=120
            )

        if resp.status_code not in [200, 201]:
            print(f"Plate Recognizer Snapshot SDK error: {resp.status_code} — {resp.text}")
            return None

        result = resp.json()

        # Debug saving is disabled by default
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # debug_filename = f"plate_recognizer_results/task_{task_id}_{timestamp}.json"
        # os.makedirs("plate_recognizer_results", exist_ok=True)
        # with open(debug_filename, 'w') as f:
        #     json.dump(result, f, indent=2)
        # print(f"💾 Saved Plate Recognizer Snapshot SDK response to: {debug_filename}")

        return result

    except Exception as e:
        print(f"Error calling Plate Recognizer Snapshot SDK: {e}")
        return None

def convert_plate_recognizer_to_label_studio(api_result, img_width, img_height):
    """Convert Plate Recognizer API response to Label Studio format (manual annotation structure)"""
    import uuid
    annotations = []
    if not api_result or 'results' not in api_result:
        return annotations

    for detection in api_result['results']:
        # --- Car region ---
        if 'vehicle' in detection and detection['vehicle']:
            vehicle = detection['vehicle']
            box = vehicle['box']
            x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            xc, yc, bw, bh = normalize(x1, y1, x2, y2, img_width, img_height)
            car_id = str(uuid.uuid4())[:10]

            # Best make/model
            car_make = "Unknown"
            car_model = "Unknown"
            if 'model_make' in detection and detection['model_make']:
                best_make_model = max(detection['model_make'], key=lambda x: x.get('score', 0))
                car_make = normalize_make(best_make_model.get('make', 'Unknown'))
                car_model = normalize_model(best_make_model.get('model', 'Unknown'))

            # Best color
            car_color = "Unknown"
            if 'color' in detection and detection['color']:
                best_color = max(detection['color'], key=lambda x: x.get('score', 0))
                car_color = normalize_color(best_color.get('color', 'Unknown'))

            # Best orientation
            car_orientation = "Unknown"
            if 'orientation' in detection and detection['orientation']:
                best_orientation = max(detection['orientation'], key=lambda x: x.get('score', 0))
                car_orientation = normalize_orientation(best_orientation.get('orientation', 'Unknown'))

            # Car type
            car_type = normalize_car_type(vehicle.get('type', 'Unknown'))

            # Rectangle region
            car_rect = {
                "id": car_id,
                "type": "rectanglelabels",
                "value": {
                    "x": float(round((xc - bw / 2) * 100, 4)),
                    "y": float(round((yc - bh / 2) * 100, 4)),
                    "width": float(round(bw * 100, 4)),
                    "height": float(round(bh * 100, 4)),
                    "rotation": 0,
                    "rectanglelabels": ["Car"]
                },
                "from_name": "labels",
                "to_name": "image"
            }
            annotations.append(car_rect)

            # Choices for car_type, car_make, car_color, car_orientation
            for from_name, value in [
                ("car_type", car_type),
                ("car_make", car_make),
                ("car_color", car_color),
                ("car_orientation", car_orientation)
            ]:
                if value != "Unknown":
                    annotations.append({
                        "id": car_id,
                        "type": "choices",
                        "value": {
                            "x": car_rect["value"]["x"],
                            "y": car_rect["value"]["y"],
                            "width": car_rect["value"]["width"],
                            "height": car_rect["value"]["height"],
                            "choices": [value],
                            "rotation": 0
                        },
                        "from_name": from_name,
                        "to_name": "image"
                    })

            # Add car_make_input and car_model_input as TextArea fields
            if car_make != "Unknown":
                annotations.append({
                    "id": car_id,
                    "type": "textarea",
                    "value": {
                        "x": car_rect["value"]["x"],
                        "y": car_rect["value"]["y"],
                        "width": car_rect["value"]["width"],
                        "height": car_rect["value"]["height"],
                        "rotation": 0,
                        "text": [car_make]
                    },
                    "from_name": "car_make_input",
                    "to_name": "image"
                })
            if car_model != "Unknown":
                annotations.append({
                    "id": car_id,
                    "type": "textarea",
                    "value": {
                        "x": car_rect["value"]["x"],
                        "y": car_rect["value"]["y"],
                        "width": car_rect["value"]["width"],
                        "height": car_rect["value"]["height"],
                        "rotation": 0,
                        "text": [car_model]
                    },
                    "from_name": "car_model_input",
                    "to_name": "image"
                })

        # --- Plate region ---
        if 'plate' in detection and detection['plate']:
            plate = detection['plate']
            box = detection['box']
            x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            xc, yc, bw, bh = normalize(x1, y1, x2, y2, img_width, img_height)
            plate_id = str(uuid.uuid4())[:10]

            # Rectangle region
            plate_rect = {
                "id": plate_id,
                "type": "rectanglelabels",
                "value": {
                    "x": float(round((xc - bw / 2) * 100, 4)),
                    "y": float(round((yc - bh / 2) * 100, 4)),
                    "width": float(round(bw * 100, 4)),
                    "height": float(round(bh * 100, 4)),
                    "rotation": 0,
                    "rectanglelabels": ["License plate"]
                },
                "from_name": "labels",
                "to_name": "image"
            }
            annotations.append(plate_rect)

            # Plate state (region)
            state_name = None
            if 'region' in detection and detection['region']:
                region_code = detection['region'].get('code', '')
                state_name = get_state_name_from_code(region_code)
                if state_name:
                    annotations.append({
                        "id": plate_id,
                        "type": "choices",
                        "value": {
                            "x": plate_rect["value"]["x"],
                            "y": plate_rect["value"]["y"],
                            "width": plate_rect["value"]["width"],
                            "height": plate_rect["value"]["height"],
                            "choices": [state_name],
                            "rotation": 0
                        },
                        "from_name": "plate_state",
                        "to_name": "image"
                    })

            # Plate number (textarea)
            annotations.append({
                "id": plate_id,
                "type": "textarea",
                "value": {
                    "x": plate_rect["value"]["x"],
                    "y": plate_rect["value"]["y"],
                    "width": plate_rect["value"]["width"],
                    "height": plate_rect["value"]["height"],
                    "rotation": 0,
                    "text": [plate]
                },
                "from_name": "plate_number",
                "to_name": "image"
            })

    return annotations

def post_annotation(task_id, annotation):
    url = f"{BASE_URL}/api/tasks/{task_id}/annotations"
    resp = requests.post(url, headers=HEADERS, json=annotation)
    if resp.ok:
        print(f"✅ Uploaded to task {task_id}")
    else:
        print(f"❌ Error for task {task_id}: {resp.status_code} — {resp.text}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process annotations for Label Studio tasks using Plate Recognizer API')
    parser.add_argument('--project-id', type=int, required=True, help='Label Studio project ID')
    parser.add_argument('--max-workers', type=int, default=7, help='Maximum number of concurrent workers (default: 10)')
    return parser.parse_args()

def validate_project_id(project_id):
    """Validate that the project ID exists in Label Studio"""
    try:
        print(f"Validating project ID: {project_id}...")
        resp = requests.get(
            f"{BASE_URL}/api/projects/{project_id}",
            headers=HEADERS,
            timeout=30
        )
        
        if resp.status_code == 404:
            print(f"Error: Project ID {project_id} not found in Label Studio")
            print("Please check that the project ID is correct and that you have access to it.")
            return False
            
        resp.raise_for_status()
        project_info = resp.json()
        print(f"Project validated: {project_info.get('title', 'Unknown project')}")
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"Error validating project: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error validating project: {e}")
        return False

def process_single_task(task, worker_id):
    """Process a single task - designed to be used with ThreadPoolExecutor"""
    try:
        task_id = task["id"]
        img_url = task["data"]["image"]
        
        # Check if task already has annotations
        if check_existing_annotations(task):
            print(f"[Worker {worker_id}] Skipping task {task_id} - already has annotations")
            return {"status": "skipped", "task_id": task_id, "reason": "already_annotated"}

        print(f"▶ [Worker {worker_id}] Processing task {task_id}")

        # Fetch and process image
        img_bytes = fetch_image(img_url, task_id)
        if not img_bytes:
            print(f"[Worker {worker_id}] Could not fetch image for task {task_id}")
            return {"status": "error", "task_id": task_id, "reason": "image_fetch_failed"}

        img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            print(f"[Worker {worker_id}] Could not decode image {img_url} for task {task_id}")
            return {"status": "error", "task_id": task_id, "reason": "image_decode_failed"}

        # Get image dimensions
        img_height, img_width = img_array.shape[:2]

        # Detect objects using Plate Recognizer API
        api_result = detect_with_plate_recognizer(img_bytes, task_id)
        if api_result:
            # Convert API result to Label Studio format
            results = convert_plate_recognizer_to_label_studio(api_result, img_width, img_height)
            if results:
                # Post annotations directly
                annotation = {"result": results}
                post_annotation(task_id, annotation)
                print(f"[Worker {worker_id}] Successfully processed task {task_id}")
                return {"status": "success", "task_id": task_id}
            else:
                print(f"[Worker {worker_id}] No objects detected in task {task_id}")
                return {"status": "success", "task_id": task_id, "reason": "no_objects"}
        else:
            print(f"[Worker {worker_id}] Failed to get results from Plate Recognizer Snapshot SDK for task {task_id}")
            return {"status": "error", "task_id": task_id, "reason": "api_failed"}
            
    except KeyError as e:
        print(f"[Worker {worker_id}] Missing data in task {task.get('id', 'unknown')}: {e}")
        return {"status": "error", "task_id": task.get('id', 'unknown'), "reason": f"missing_data: {e}"}
    except Exception as e:
        print(f"[Worker {worker_id}] Error processing task {task.get('id', 'unknown')}: {e}")
        return {"status": "error", "task_id": task.get('id', 'unknown'), "reason": f"exception: {e}"}

def main():
    # Parse command line arguments
    args = parse_arguments()
    global PROJECT_ID
    PROJECT_ID = args.project_id
    
    print(f"Starting annotation process for project ID: {PROJECT_ID}")
    print(f"Using Plate Recognizer Snapshot SDK (Docker) for detection")
    
    # Validate project ID before proceeding
    if not validate_project_id(PROJECT_ID):
        print("Aborting due to invalid project ID")
        sys.exit(1)
    
    # Fetch tasks
    try:
        tasks = fetch_tasks()
        if not tasks:
            print("No tasks found in the project. Please check if the project contains any tasks.")
            return
        print(f"Found {len(tasks)} tasks")
    except Exception as e:
        print(f"Error fetching tasks: {e}")
        sys.exit(1)

    # Track statistics
    skipped_tasks = 0
    processed_tasks = 0
    error_tasks = 0

    # Determine optimal number of workers based on user preference and system capabilities
    max_workers = min(args.max_workers, len(tasks))  # Don't exceed the number of tasks
    print(f"Using {max_workers} concurrent workers for processing")

    # Start timing for processing window
    processing_start_time = time.perf_counter()

    # Process tasks concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor with worker IDs
        future_to_task = {}
        for i, task in enumerate(tasks):
            worker_id = i % max_workers + 1
            future = executor.submit(process_single_task, task, worker_id)
            future_to_task[future] = task
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result["status"] == "success":
                    processed_tasks += 1
                elif result["status"] == "skipped":
                    skipped_tasks += 1
                else:
                    error_tasks += 1
                    
                # Print progress
                total_completed = processed_tasks + skipped_tasks + error_tasks
                progress_percent = (total_completed / len(tasks)) * 100
                print(f"Progress: {total_completed}/{len(tasks)} ({progress_percent:.1f}%) "
                      f"({processed_tasks}, {skipped_tasks},  {error_tasks})")
                
            except Exception as e:
                print(f"Unexpected error processing task {task.get('id', 'unknown')}: {e}")
                error_tasks += 1

    # End timing
    processing_elapsed = time.perf_counter() - processing_start_time

    print(f"\n Processing complete!")
    print(f"Final Statistics:")
    print(f"   • Total tasks: {len(tasks)}")
    print(f"   • Successfully processed: {processed_tasks}")
    print(f"   • Skipped (already annotated): {skipped_tasks}")
    print(f"   • Errors: {error_tasks}")
    print(f"   • Concurrent workers used: {max_workers}")

    # Throughput metrics
    total_done = processed_tasks + skipped_tasks + error_tasks
    processed_attempts = processed_tasks + error_tasks  # attempted annotations (excludes skipped)
    print(f"Total processing time: {processing_elapsed:.2f} seconds")
    if total_done:
        print(f"Throughput (all tasks): {total_done/processing_elapsed:.2f} tasks/sec")
    if processed_attempts:
        print(f"Throughput (annotate attempts): {processed_attempts/processing_elapsed:.2f} tasks/sec")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Error: {e}")
        sys.exit(1)