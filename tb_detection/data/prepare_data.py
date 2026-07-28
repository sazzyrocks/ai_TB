import os
import sys
import shutil
import argparse
from sklearn.model_selection import train_test_split

# Add parent directory to path so config can be imported when running script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def parse_list_file(list_path: str):
    """
    Parses a TBX11K list file. Returns lists of image relative paths and labels.
    Binary classification:
      - 'tb/' images are Class 1 (Tuberculosis)
      - 'health/' and 'sick/' images are Class 0 (Normal)
    """
    image_paths = []
    labels = []
    
    with open(list_path, "r") as f:
        for line in f:
            rel_path = line.strip()
            if not rel_path:
                continue
            
            # Categorize based on folder name
            if rel_path.startswith("tb/"):
                label = 1  # Tuberculosis
            elif rel_path.startswith("health/") or rel_path.startswith("sick/"):
                label = 0  # Normal
            else:
                continue  # Skip other patterns
                
            image_paths.append(rel_path)
            labels.append(label)
            
    return image_paths, labels

def copy_images(image_paths, labels, source_base_dir, dest_split_dir):
    """
    Copies images from TBX11K imgs directory to the processed split directory.
    """
    for rel_path, label in zip(image_paths, labels):
        src_path = os.path.join(source_base_dir, rel_path)
        if not os.path.exists(src_path):
            print(f"Warning: Source file not found: {src_path}")
            continue
            
        class_folder = Config.CLASS_NAMES[label]
        dest_dir = os.path.join(dest_split_dir, class_folder)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy to destination (retaining base name)
        dest_path = os.path.join(dest_dir, os.path.basename(rel_path))
        shutil.copy2(src_path, dest_path)

def prepare_tbx11k_data(tbx11k_dir: str, processed_dir: str):
    print("Preparing TBX11K dataset splits...")
    
    # Paths to lists and images
    lists_dir = os.path.join(tbx11k_dir, "lists")
    imgs_dir = os.path.join(tbx11k_dir, "imgs")
    
    train_list_path = os.path.join(lists_dir, "TBX11K_train.txt")
    val_list_path = os.path.join(lists_dir, "TBX11K_val.txt")
    
    if not os.path.exists(train_list_path) or not os.path.exists(val_list_path):
        raise FileNotFoundError(f"TBX11K lists not found in: {lists_dir}")
    if not os.path.exists(imgs_dir):
        raise FileNotFoundError(f"TBX11K images directory not found: {imgs_dir}")
        
    # 1. Parse lists
    print("Parsing train list...")
    train_paths, train_labels = parse_list_file(train_list_path)
    
    print("Parsing val/test list...")
    val_raw_paths, val_raw_labels = parse_list_file(val_list_path)
    
    # 2. Stratified split of validation list into validation and test sets (50/50 split)
    print("Performing stratified split on validation list to create val and test sets...")
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        val_raw_paths, val_raw_labels, test_size=0.5, stratify=val_raw_labels, random_state=42
    )
    
    print(f"Data Distribution:")
    print(f"  - Train: {len(train_paths)} images (Normal={train_labels.count(0)}, TB={train_labels.count(1)})")
    print(f"  - Val:   {len(val_paths)} images (Normal={val_labels.count(0)}, TB={val_labels.count(1)})")
    print(f"  - Test:  {len(test_paths)} images (Normal={test_labels.count(0)}, TB={test_labels.count(1)})")
    
    # 3. Copy files to structured subdirectories
    print("\nCopying training set images...")
    copy_images(train_paths, train_labels, imgs_dir, os.path.join(processed_dir, "train"))
    
    print("Copying validation set images...")
    copy_images(val_paths, val_labels, imgs_dir, os.path.join(processed_dir, "val"))
    
    print("Copying test set images...")
    copy_images(test_paths, test_labels, imgs_dir, os.path.join(processed_dir, "test"))
    
    print(f"\nTBX11K organization complete! Processed dataset splits stored in: {processed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and organize TBX11K dataset splits.")
    parser.add_argument("--tbx11k_dir", type=str, default=os.path.join(Config.BASE_DIR, "TBX11K"), help="Path to TBX11K dataset root")
    parser.add_argument("--processed_dir", type=str, default=Config.PROCESSED_DATA_DIR, help="Path to save organized splits")
    args = parser.parse_args()
    
    prepare_tbx11k_data(tbx11k_dir=args.tbx11k_dir, processed_dir=args.processed_dir)
