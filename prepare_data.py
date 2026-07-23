import os
import zipfile
from pathlib import Path

def setup_dataset_structure():
    base_dir = Path("data/chest_xray")
    splits = ["train", "val", "test"]
    classes = ["NORMAL", "PNEUMONIA"]
    
    for split in splits:
        for cls in classes:
            os.makedirs(base_dir / split / cls, exist_ok=True)
            
    print(f"✅ Directory structure created at: {base_dir}")
    print("\n--- Manual Download Instructions ---")
    print("If you do not have the Kaggle API configured:")
    print("1. Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle.")
    print("2. Extract and place your images into the respective folders:")
    print(f"   - {base_dir}/train/NORMAL & PNEUMONIA")
    print(f"   - {base_dir}/val/NORMAL & PNEUMONIA")
    print(f"   - {base_dir}/test/NORMAL & PNEUMONIA")

if __name__ == "__main__":
    setup_dataset_structure()