import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "annotations.json")
bbox_output_path = os.path.join(current_dir, "make_car_plate.txt")
plate_output_path = os.path.join(current_dir, "char_state.txt")

with open(json_path, 'r') as f:
    data = json.load(f)

bbox_lines = []
plate_lines = []

for task in data:
    raw_image_path = task["data"]["image"]
    file_name = raw_image_path.split("/")[-1]

    if "-image_" in file_name:
        file_name = file_name.split("-image_")[-1]
        file_name = "image_" + file_name

    image_bbox_path = f"images/origin_image/{file_name}"
    image_plate_path = f"images/plate_image/{file_name}"

    results = task["annotations"][0]["result"]

    for res in results:
        if res["type"] == "rectanglelabels":
            label = res["value"]["rectanglelabels"][0].lower()
            x = res["value"]["x"]
            y = res["value"]["y"]
            w = res["value"]["width"]
            h = res["value"]["height"]
            img_w = res["original_width"]
            img_h = res["original_height"]

            abs_x = x * img_w / 100
            abs_y = y * img_h / 100
            abs_w = w * img_w / 100
            abs_h = h * img_h / 100

            x_center = (abs_x + abs_w / 2) / img_w
            y_center = (abs_y + abs_h / 2) / img_h
            norm_w = abs_w / img_w
            norm_h = abs_h / img_h

            class_id = 0 if label == "car" else 1 if label == "license plate" else -1
            if class_id == -1:
                continue

            line = f"{image_bbox_path} {class_id} 0 {x_center:.9f} {y_center:.9f} {norm_w:.9f} {norm_h:.9f}"
            bbox_lines.append(line)

        elif res["type"] == "textarea" and res["from_name"] == "plate_number":
            plate_text = res["value"]["text"][0].strip()
            line = f"{image_plate_path} {plate_text} 30"
            plate_lines.append(line)

with open(bbox_output_path, "w") as f:
    for line in bbox_lines:
        f.write(line + "\n")

with open(plate_output_path, "w") as f:
    for line in plate_lines:
        f.write(line + "\n")

print("✅ Done!")
print(f"Bounding boxes saved to: {bbox_output_path}")
print(f"Plate texts saved to: {plate_output_path}")
