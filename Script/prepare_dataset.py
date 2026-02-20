import os
import shutil
import re
from pathlib import Path
import random

"""
Dataset Preparation Script for YOLO Training
This script matches Label Studio exported labels with images and organizes them into YOLO format.

Label Studio exports labels with a UUID prefix like:
  - Label: 02263b54__[JZ-NRPWPA10044][IMAGEID-019c2a96c1597e6ab24675147c9d7af9].txt
  - Image: [JZ-NRPWPA10044][IMAGEID-019c2a96c1597e6ab24675147c9d7af9].jpg

This script:
1. Matches labels to images by removing the UUID prefix
2. Renames labels to match images exactly (without UUID prefix)
3. Splits data into train/val sets (80/20 split)
4. Creates proper YOLO directory structure
5. Generates data.yaml configuration file
"""

# Configuration
EXPORT_DIR = "export_227629_project-227629-at-2026-02-12-10-40-f229e6cc"
OUTPUT_DIR = "dataset"
TRAIN_SPLIT = 0.8  # 80% train, 20% validation

# Class mapping from Label Studio to YOLO format
# Label Studio exported classes (you may need to adjust based on your actual classes)
CLASS_MAPPING = {
    14: "Car",           # Update these numbers based on your Label Studio export
    15: "License-Plate"  # Check the actual class IDs in your label files
}

def extract_base_name(filename):
    """
    Extract the base image name from a label filename by removing UUID prefix.
    Example: '02263b54__[JZ-NRPWPA10044][IMAGEID-019c2a96c1597e6ab24675147c9d7af9].txt'
         -> '[JZ-NRPWPA10044][IMAGEID-019c2a96c1597e6ab24675147c9d7af9]'
    """
    # Remove UUID prefix (8 hex chars + __)
    match = re.search(r'^[0-9a-f]{8}__(.+)$', filename)
    if match:
        base_with_ext = match.group(1)
        # Remove extension
        base_name = os.path.splitext(base_with_ext)[0]
        return base_name
    else:
        # No UUID prefix, just remove extension
        return os.path.splitext(filename)[0]

def create_directory_structure():
    """Create YOLO dataset directory structure"""
    dirs = [
        os.path.join(OUTPUT_DIR, "images", "train"),
        os.path.join(OUTPUT_DIR, "images", "val"),
        os.path.join(OUTPUT_DIR, "labels", "train"),
        os.path.join(OUTPUT_DIR, "labels", "val"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✅ Created directory structure in '{OUTPUT_DIR}'")

def match_labels_to_images():
    """
    Match label files to image files by extracting base names.
    Returns a dictionary mapping base_name -> (image_path, label_path)
    """
    images_dir = os.path.join(EXPORT_DIR, "images")
    labels_dir = os.path.join(EXPORT_DIR, "labels")
    
    # Build image mapping: base_name -> full_image_path
    image_map = {}
    for img_file in os.listdir(images_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_name = os.path.splitext(img_file)[0]
            image_map[base_name] = os.path.join(images_dir, img_file)
    
    print(f"📊 Found {len(image_map)} images")
    
    # Match labels to images
    matched_pairs = {}
    unmatched_labels = []
    
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            base_name = extract_base_name(label_file)
            
            if base_name in image_map:
                label_path = os.path.join(labels_dir, label_file)
                matched_pairs[base_name] = {
                    'image': image_map[base_name],
                    'label': label_path
                }
            else:
                unmatched_labels.append(label_file)
    
    print(f"✅ Matched {len(matched_pairs)} label-image pairs")
    if unmatched_labels:
        print(f"⚠️  {len(unmatched_labels)} labels could not be matched to images")
        print(f"   First few unmatched: {unmatched_labels[:5]}")
    
    return matched_pairs

def remap_class_ids(label_path):
    """
    Read label file and remap class IDs from Label Studio to YOLO format (0-indexed).
    Label Studio might export class 14, 15, etc. We need 0, 1 for YOLO.
    
    Returns list of lines with remapped class IDs.
    """
    remapped_lines = []
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                original_class = int(parts[0])
                
                # Remap class ID to 0-indexed
                if original_class == 14:
                    new_class = 0  # Car
                elif original_class == 15:
                    new_class = 1  # License Plate
                else:
                    print(f"⚠️  Unknown class ID {original_class} in {label_path}")
                    continue
                
                # Reconstruct line with new class ID
                remapped_line = f"{new_class} {' '.join(parts[1:])}"
                remapped_lines.append(remapped_line)
    
    return remapped_lines

def split_and_organize_dataset(matched_pairs):
    """
    Split matched pairs into train/val sets and copy to dataset directory.
    """
    # Convert to list and shuffle
    pairs_list = list(matched_pairs.items())
    random.shuffle(pairs_list)
    
    # Calculate split point
    split_idx = int(len(pairs_list) * TRAIN_SPLIT)
    train_pairs = pairs_list[:split_idx]
    val_pairs = pairs_list[split_idx:]
    
    print(f"\n📦 Dataset split:")
    print(f"   Training: {len(train_pairs)} samples")
    print(f"   Validation: {len(val_pairs)} samples")
    
    def copy_pairs(pairs, split_name):
        """Copy image-label pairs to train or val directory"""
        for base_name, paths in pairs:
            img_src = paths['image']
            label_src = paths['label']
            
            # Destination paths (using base_name without UUID prefix)
            img_ext = os.path.splitext(img_src)[1]
            img_dst = os.path.join(OUTPUT_DIR, "images", split_name, f"{base_name}{img_ext}")
            label_dst = os.path.join(OUTPUT_DIR, "labels", split_name, f"{base_name}.txt")
            
            # Copy image
            shutil.copy2(img_src, img_dst)
            
            # Process and copy label with remapped class IDs
            remapped_lines = remap_class_ids(label_src)
            with open(label_dst, 'w') as f:
                f.write('\n'.join(remapped_lines) + '\n')
    
    print(f"\n📋 Copying files...")
    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    print(f"✅ Dataset organized!")

def create_data_yaml():
    """Create data.yaml configuration file for YOLO training"""
    dataset_path = os.path.abspath(OUTPUT_DIR)
    
    yaml_content = f"""# YOLO Dataset Configuration
# Generated by prepare_dataset.py

# Dataset paths (absolute paths recommended)
path: {dataset_path}
train: images/train
val: images/val

# Number of classes
nc: 2

# Class names
names:
  0: Car
  1: License-Plate
"""
    
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created {yaml_path}")

def main():
    print("=" * 60)
    print("YOLO Dataset Preparation Script")
    print("=" * 60)
    
    # Check if export directory exists
    if not os.path.exists(EXPORT_DIR):
        print(f"❌ Export directory '{EXPORT_DIR}' not found!")
        print(f"   Please make sure the Label Studio export folder is in the project root.")
        return
    
    # Step 1: Create directory structure
    print("\n[1/4] Creating directory structure...")
    create_directory_structure()
    
    # Step 2: Match labels to images
    print("\n[2/4] Matching labels to images...")
    matched_pairs = match_labels_to_images()
    
    if not matched_pairs:
        print("❌ No matched pairs found! Cannot proceed.")
        return
    
    # Step 3: Split and organize dataset
    print("\n[3/4] Organizing dataset...")
    split_and_organize_dataset(matched_pairs)
    
    # Step 4: Create data.yaml
    print("\n[4/4] Creating configuration file...")
    create_data_yaml()
    
    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print(f"📁 Dataset location: {os.path.abspath(OUTPUT_DIR)}")
    print("\nNext steps:")
    print("  1. Review the dataset structure in 'dataset/' folder")
    print("  2. Run 'python Script/train_yolo.py' to start training")
    print("=" * 60)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
