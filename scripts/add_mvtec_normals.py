import shutil
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

# ================= CONFIG =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MVTEC_GOOD = PROJECT_ROOT / "data" / "raw" / "mvtec_transistor" / "train" / "good"
OUT_IMG_DIR = PROJECT_ROOT / "data" / "images" / "train"
OUT_ANN_DIR = PROJECT_ROOT / "data" / "annotations" / "train"
# ==========================================

print("="*60)
print("ADDING MVTEC NORMAL IMAGES TO TRAINING")
print("="*60)

if not MVTEC_GOOD.exists():
    print(f"\n❌ ERROR: MVTec good folder not found!")
    print(f"Expected: {MVTEC_GOOD}")
    exit(1)

print(f"\nMVTec folder: {MVTEC_GOOD}")
print(f"Output images: {OUT_IMG_DIR}")
print(f"Output annotations: {OUT_ANN_DIR}")

# Get all MVTec good images
mvtec_images = list(MVTEC_GOOD.glob("*.png"))
print(f"\nFound {len(mvtec_images)} MVTec normal images")

if len(mvtec_images) == 0:
    print("❌ No images found!")
    exit(1)

def create_empty_voc_xml(img_path, save_path):
    """Create empty VOC XML (no defects)"""
    img = Image.open(img_path)
    width, height = img.size

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = img_path.name

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"  # RGB

    # No objects = defect-free
    
    tree = ET.ElementTree(annotation)
    tree.write(save_path)

count = 0

print("\nAdding images...\n")

for img_path in mvtec_images:
    # Create new filename (mvtec_XXX.png)
    new_name = f"mvtec_{img_path.stem}.png"
    
    img_dst = OUT_IMG_DIR / new_name
    ann_dst = OUT_ANN_DIR / f"mvtec_{img_path.stem}.xml"
    
    # Copy image
    shutil.copy(img_path, img_dst)
    
    # Create empty XML
    create_empty_voc_xml(img_path, ann_dst)
    
    count += 1
    
    if count % 20 == 0:
        print(f"✓ Added {count}/{len(mvtec_images)}")

print("\n" + "="*60)
print(f"✅ Successfully added {count} MVTec normal images")
print("="*60)
print("\nNext step: Retrain the model")
print("Command: python FINAL_train.py")