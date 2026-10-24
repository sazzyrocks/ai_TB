import os
import sys
import shutil
import zipfile
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add parent directory to path so config can be imported when running script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def parse_lines(lines):
    """
    Parses TBX11K lines into relative image paths and binary labels:
      - 'tb/' images are Class 1 (Tuberculosis)
      - 'health/' and 'sick/' images are Class 0 (Normal)
    """
    image_paths = []
    labels = []
    for line in lines:
        rel_path = line.strip()
        if not rel_path:
            continue
        if rel_path.startswith("tb/"):
            label = 1  # Tuberculosis
        elif rel_path.startswith("health/") or rel_path.startswith("sick/"):
            label = 0  # Normal
        else:
            continue
        image_paths.append(rel_path)
        labels.append(label)
    return image_paths, labels

def parse_list_file(list_path: str):
    with open(list_path, "r", encoding="utf-8") as f:
        return parse_lines(f.readlines())

def extract_from_zip(zip_file: zipfile.ZipFile, image_paths, labels, dest_split_dir):
    """
    Extracts images directly from the TBX11K zip file to the destination split folder.
    """
    for rel_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), ncols=100):
        zip_img_path = f"imgs/{rel_path}"
        class_folder = Config.CLASS_NAMES[label]
        dest_dir = os.path.join(dest_split_dir, class_folder)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(rel_path))
        
        try:
            with zip_file.open(zip_img_path) as source, open(dest_path, "wb") as target:
                shutil.copyfileobj(source, target)
        except KeyError:
            print(f"Warning: File not found in zip: {zip_img_path}")

def copy_images(image_paths, labels, source_base_dir, dest_split_dir):
    """
    Copies images from unzipped TBX11K imgs directory to the processed split directory.
    """
    for rel_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), ncols=100):
        src_path = os.path.join(source_base_dir, rel_path)
        if not os.path.exists(src_path):
            print(f"Warning: Source file not found: {src_path}")
            continue
            
        class_folder = Config.CLASS_NAMES[label]
        dest_dir = os.path.join(dest_split_dir, class_folder)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(rel_path))
        shutil.copy2(src_path, dest_path)

def prepare_tbx11k_data(source_path: str, processed_dir: str, limit_train: int = None, limit_val: int = None, limit_test: int = None):
    print(f"Preparing TBX11K dataset splits from: {source_path}")
    
    is_zip = zipfile.is_zipfile(source_path)
    
    if is_zip:
        print("Detected ZIP archive. Processing directly from archive...")
        with zipfile.ZipFile(source_path, "r") as zf:
            train_raw_lines = zf.read("lists/TBX11K_train.txt").decode("utf-8").splitlines()
            val_raw_lines = zf.read("lists/TBX11K_val.txt").decode("utf-8").splitlines()
            
            train_paths, train_labels = parse_lines(train_raw_lines)
            val_raw_paths, val_raw_labels = parse_lines(val_raw_lines)
            
            # Stratified split of val list into validation and test sets (50/50 split)
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                val_raw_paths, val_raw_labels, test_size=0.5, stratify=val_raw_labels, random_state=42
            )
            
            if limit_train:
                train_paths, train_labels = train_paths[:limit_train], train_labels[:limit_train]
            if limit_val:
                val_paths, val_labels = val_paths[:limit_val], val_labels[:limit_val]
            if limit_test:
                test_paths, test_labels = test_paths[:limit_test], test_labels[:limit_test]
                
            print(f"Dataset Distribution:")
            print(f"  - Train: {len(train_paths)} images (Normal={train_labels.count(0)}, TB={train_labels.count(1)})")
            print(f"  - Val:   {len(val_paths)} images (Normal={val_labels.count(0)}, TB={val_labels.count(1)})")
            print(f"  - Test:  {len(test_paths)} images (Normal={test_labels.count(0)}, TB={test_labels.count(1)})")
            
            # Remove old processed splits if needed
            for split_name in ["train", "val", "test"]:
                split_path = os.path.join(processed_dir, split_name)
                if os.path.exists(split_path):
                    shutil.rmtree(split_path)
            
            print("\nExtracting training set...")
            extract_from_zip(zf, train_paths, train_labels, os.path.join(processed_dir, "train"))
            
            print("Extracting validation set...")
            extract_from_zip(zf, val_paths, val_labels, os.path.join(processed_dir, "val"))
            
            print("Extracting test set...")
            extract_from_zip(zf, test_paths, test_labels, os.path.join(processed_dir, "test"))
    else:
        lists_dir = os.path.join(source_path, "lists")
        imgs_dir = os.path.join(source_path, "imgs")
        train_list_path = os.path.join(lists_dir, "TBX11K_train.txt")
        val_list_path = os.path.join(lists_dir, "TBX11K_val.txt")
        
        if not os.path.exists(train_list_path) or not os.path.exists(val_list_path):
            raise FileNotFoundError(f"TBX11K lists not found in: {lists_dir}")
        if not os.path.exists(imgs_dir):
            raise FileNotFoundError(f"TBX11K images directory not found: {imgs_dir}")
            
        train_paths, train_labels = parse_list_file(train_list_path)
        val_raw_paths, val_raw_labels = parse_list_file(val_list_path)
        
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            val_raw_paths, val_raw_labels, test_size=0.5, stratify=val_raw_labels, random_state=42
        )
        
        if limit_train:
            train_paths, train_labels = train_paths[:limit_train], train_labels[:limit_train]
        if limit_val:
            val_paths, val_labels = val_paths[:limit_val], val_labels[:limit_val]
        if limit_test:
            test_paths, test_labels = test_paths[:limit_test], test_labels[:limit_test]
            
        print(f"Dataset Distribution:")
        print(f"  - Train: {len(train_paths)} images (Normal={train_labels.count(0)}, TB={train_labels.count(1)})")
        print(f"  - Val:   {len(val_paths)} images (Normal={val_labels.count(0)}, TB={val_labels.count(1)})")
        print(f"  - Test:  {len(test_paths)} images (Normal={test_labels.count(0)}, TB={test_labels.count(1)})")
        
        for split_name in ["train", "val", "test"]:
            split_path = os.path.join(processed_dir, split_name)
            if os.path.exists(split_path):
                shutil.rmtree(split_path)
                
        print("\nCopying training set...")
        copy_images(train_paths, train_labels, imgs_dir, os.path.join(processed_dir, "train"))
        
        print("Copying validation set...")
        copy_images(val_paths, val_labels, imgs_dir, os.path.join(processed_dir, "val"))
        
        print("Copying test set...")
        copy_images(test_paths, test_labels, imgs_dir, os.path.join(processed_dir, "test"))
        
    print(f"\nTBX11K dataset preparation complete! Splits saved to: {processed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and organize TBX11K dataset splits.")
    parser.add_argument(
        "--source", 
        type=str, 
        default=os.path.join(os.path.expanduser("~"), "Downloads", "TBX11K.zip"),
        help="Path to TBX11K folder or .zip file"
    )
    parser.add_argument("--processed_dir", type=str, default=Config.PROCESSED_DATA_DIR, help="Path to save organized splits")
    parser.add_argument("--limit_train", type=int, default=None, help="Optional limit for training images")
    parser.add_argument("--limit_val", type=int, default=None, help="Optional limit for validation images")
    parser.add_argument("--limit_test", type=int, default=None, help="Optional limit for test images")
    args = parser.parse_args()
    
    prepare_tbx11k_data(
        source_path=args.source,
        processed_dir=args.processed_dir,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        limit_test=args.limit_test
    )
