"""
Reduces the PlantifyDr dataset to 30% and creates a smaller zip for uploading.
Run this script once locally, then upload the output zip to Google Drive.
"""

import os
import shutil
import random
import zipfile
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
SOURCE_DIR  = Path(r"C:\Users\USER\Downloads\PlantDiseaseProject\dataset")  # extracted dataset
OUTPUT_DIR  = Path(r"C:\Users\USER\Downloads\PlantDiseaseProject\dataset_reduced")
OUTPUT_ZIP  = Path(r"C:\Users\USER\Downloads\PlantDiseasesDataset_reduced.zip")
FRACTION    = 0.50   # keep 50% of images per class
RANDOM_SEED = 42
# ────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

if not SOURCE_DIR.exists():
    # Try the flat dataset if already restructured
    SOURCE_DIR = Path(r"C:\Users\USER\Downloads\PlantDiseaseProject\dataset_flat")
    if not SOURCE_DIR.exists():
        print("❌ Dataset folder not found!")
        print(f"   Expected at: {SOURCE_DIR}")
        print("   Please extract PlantDiseasesDataset.zip first.")
        exit(1)

print(f"📂 Source : {SOURCE_DIR}")
print(f"📂 Output : {OUTPUT_DIR}")
print(f"📊 Keeping: {int(FRACTION*100)}% of images per class")
print()

if OUTPUT_DIR.exists():
    print("🗑  Removing old reduced dataset...")
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)

total_copied = 0
total_skipped = 0

# Walk through every folder looking for train/valid splits
for root, dirs, files in os.walk(SOURCE_DIR):
    root_path = Path(root)
    images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not images:
        continue

    # Select a random fraction
    n_keep = max(1, int(len(images) * FRACTION))
    selected = random.sample(images, n_keep)
    total_skipped += len(images) - n_keep

    # Mirror the folder structure inside OUTPUT_DIR
    rel_path = root_path.relative_to(SOURCE_DIR)
    dest_dir = OUTPUT_DIR / rel_path
    dest_dir.mkdir(parents=True, exist_ok=True)

    for img in selected:
        shutil.copy2(root_path / img, dest_dir / img)
        total_copied += 1

    print(f"  ✅ {rel_path}  →  {n_keep}/{len(images)} images kept")

print()
print(f"📊 Total copied : {total_copied:,} images")
print(f"📊 Total skipped: {total_skipped:,} images")
print()

# Zip the reduced dataset
print(f"📦 Creating zip: {OUTPUT_ZIP}")
with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for file_path in OUTPUT_DIR.rglob('*'):
        if file_path.is_file():
            zf.write(file_path, file_path.relative_to(OUTPUT_DIR.parent))

zip_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
print(f"✅ Done! Zip size: {zip_size_mb:.1f} MB")
print(f"📁 Saved to: {OUTPUT_ZIP}")
print()
print("Next step: Upload this zip to Google Drive, then use it in Colab!")
