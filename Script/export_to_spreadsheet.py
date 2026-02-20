import argparse
import csv
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


def _safe_basename(url: str) -> str:
    if not url:
        return ""
    url = re.split(r"[?#]", url, maxsplit=1)[0]
    if url.startswith("s3://"):
        return url.rstrip("/").split("/")[-1]
    return os.path.basename(url.rstrip("/"))


def _load_json_records(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
                return
            if isinstance(data, dict):
                yield data
                return
        except json.JSONDecodeError:
            pass

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError:
                continue


def _get_annotation_results(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    annotations = record.get("annotations") or []
    if isinstance(annotations, list) and annotations:
        for ann in reversed(annotations):
            if isinstance(ann, dict) and isinstance(ann.get("result"), list):
                return ann["result"]
    predictions = record.get("predictions") or []
    if isinstance(predictions, list) and predictions:
        for pred in reversed(predictions):
            if isinstance(pred, dict) and isinstance(pred.get("result"), list):
                return pred["result"]
    return []


def _extract_values_by_id(results: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    cars: Dict[str, Dict[str, Any]] = {}
    plates: Dict[str, Dict[str, Any]] = {}

    for r in results:
        if not isinstance(r, dict):
            continue
        r_type = r.get("type")
        rid = str(r.get("id")) if r.get("id") is not None else None
        value = r.get("value") or {}
        if r_type == "rectanglelabels" and rid and isinstance(value, dict):
            labels = value.get("rectanglelabels") or []
            if labels and isinstance(labels, list):
                label = str(labels[0]).strip().lower()
                if label == "car":
                    cars[rid] = {
                        "car_type": None,
                        "car_make": None,
                        "car_model": None,
                        "car_color": None,
                        "car_orientation": None,
                    }
                elif label == "license plate":
                    plates[rid] = {
                        "plate_state": None,
                        "plate_number": None,
                    }

    for r in results:
        if not isinstance(r, dict):
            continue
        r_type = r.get("type")
        rid = str(r.get("id")) if r.get("id") is not None else None
        value = r.get("value") or {}
        from_name = r.get("from_name")

        if rid in cars:
            if r_type == "choices" and isinstance(value, dict):
                choices = value.get("choices") or []
                val = str(choices[0]).strip() if choices else None
                if from_name == "car_type":
                    cars[rid]["car_type"] = cars[rid]["car_type"] or val
                elif from_name == "car_make":
                    cars[rid]["car_make"] = cars[rid]["car_make"] or val
                elif from_name == "car_color":
                    cars[rid]["car_color"] = cars[rid]["car_color"] or val
                elif from_name == "car_orientation":
                    cars[rid]["car_orientation"] = cars[rid]["car_orientation"] or val
            elif r_type == "textarea" and isinstance(value, dict):
                text = value.get("text") or []
                val = str(text[0]).strip() if text else None
                if from_name == "car_make_input" and val:
                    cars[rid]["car_make"] = cars[rid]["car_make"] or val
                elif from_name == "car_model_input" and val:
                    cars[rid]["car_model"] = cars[rid]["car_model"] or val

        if rid in plates:
            if r_type == "choices" and isinstance(value, dict):
                choices = value.get("choices") or []
                val = str(choices[0]).strip() if choices else None
                if from_name == "plate_state":
                    plates[rid]["plate_state"] = plates[rid]["plate_state"] or val
            elif r_type == "textarea" and isinstance(value, dict):
                text = value.get("text") or []
                val = str(text[0]).strip() if text else None
                if from_name == "plate_number":
                    plates[rid]["plate_number"] = plates[rid]["plate_number"] or val

    return cars, plates


def _extract_row(record: Dict[str, Any]) -> Dict[str, Any]:
    data = record.get("data") or {}
    image_url = data.get("image") or ""
    image_name = _safe_basename(image_url)

    results = _get_annotation_results(record)
    cars, plates = _extract_values_by_id(results)

    car_vals = next(iter(cars.values()), {
        "car_type": None,
        "car_make": None,
        "car_model": None,
        "car_color": None,
        "car_orientation": None,
    })
    plate_vals = next(iter(plates.values()), {
        "plate_state": None,
        "plate_number": None,
    })

    return {
        "task_id": record.get("id"),
        "image_url": image_url,
        "image_name": image_name,
        "car_type": car_vals.get("car_type"),
        "car_make": car_vals.get("car_make"),
        "car_model": car_vals.get("car_model"),
        "car_color": car_vals.get("car_color"),
        "car_orientation": car_vals.get("car_orientation"),
        "plate_state": plate_vals.get("plate_state"),
        "plate_number": plate_vals.get("plate_number"),
    }


def write_csv(rows: Iterable[Dict[str, Any]], output_path: str) -> None:
    fieldnames = [
        "task_id",
        "image_url",
        "image_name",
        "car_type",
        "car_make",
        "car_model",
        "car_color",
        "car_orientation",
        "plate_state",
        "plate_number",
    ]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description="Export car and plate attributes from Label Studio JSON to CSV")
    parser.add_argument("--input", required=True, help="Path to Label Studio export JSON file")
    parser.add_argument("--output", default="exported_attributes.csv", help="Output CSV path (openable in Excel)")
    args = parser.parse_args()

    records = list(_load_json_records(args.input))
    rows = (_extract_row(rec) for rec in records)
    write_csv(rows, args.output)
    print(f"Saved {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()


