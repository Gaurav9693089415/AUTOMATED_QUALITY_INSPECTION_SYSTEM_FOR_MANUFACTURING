import shutil
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

# ================= CONFIG =================
SCRIPT_PATH = Path(__file__).resolve()

if SCRIPT_PATH.parent.name == "scripts":
    PROJECT_ROOT = SCRIPT_PATH.parent.parent
else:
    PROJECT_ROOT = SCRIPT_PATH.parent

RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "DeepPCB-master" / "PCBData"
TRAINVAL_FILE = RAW_ROOT / "trainval.txt"

OUT_IMG_DIR = PROJECT_ROOT / "data" / "images" / "train"
OUT_ANN_DIR = PROJECT_ROOT / "data" / "annotations" / "train"

MAX_DEFECTIVE = 1500  # All defective images
MAX_NORMAL = 1501     # All normal images
# ==========================================

print("Checking paths...")
print(f"Project root: {PROJECT_ROOT}")
print(f"Raw data: {RAW_ROOT}")
print(f"Trainval file exists: {TRAINVAL_FILE.exists()}")

if not TRAINVAL_FILE.exists():
    print("\n❌ ERROR: trainval.txt not found!")
    exit(1)

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_ANN_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    "1": "open",
    "2": "short",
    "3": "mousebite",
    "4": "spur",
    "5": "copper",
    "6": "pin-hole"
}

def create_voc_xml(img_path, ann_path, save_path, is_normal=False):
    """Create VOC format XML from DeepPCB annotation"""
    img = Image.open(img_path)
    width, height = img.size

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = img_path.name

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "1"

    if is_normal:
        # Normal image - no defects, empty XML
        tree = ET.ElementTree(annotation)
        tree.write(save_path)
        return True, 0
    
    # Defective image - parse annotations
    defect_count = 0
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()  # Space-separated
            
            if len(parts) != 5:
                continue

            x1, y1, x2, y2, cls_id = parts
            if cls_id not in CLASS_MAP:
                continue

            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = CLASS_MAP[cls_id]

            bbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bbox, "xmin").text = x1
            ET.SubElement(bbox, "ymin").text = y1
            ET.SubElement(bbox, "xmax").text = x2
            ET.SubElement(bbox, "ymax").text = y2
            
            defect_count += 1

    if defect_count > 0:
        tree = ET.ElementTree(annotation)
        tree.write(save_path)
        return True, defect_count
    
    return False, 0

defective_count = 0
normal_count = 0
skipped = 0

print("\nStarting dataset preparation...\n")

with open(TRAINVAL_FILE, "r") as f:
    lines = f.readlines()
    total_lines = len(lines)
    
    for line_num, line in enumerate(lines, 1):
        if defective_count >= MAX_DEFECTIVE and normal_count >= MAX_NORMAL:
            break

        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) != 2:
            continue
            
        img_rel, ann_rel = parts

        img_path_base = RAW_ROOT / img_rel
        ann_path = RAW_ROOT / ann_rel
        
        img_dir = img_path_base.parent
        img_stem = img_path_base.stem
        
        # Process DEFECTIVE images (_test)
        if defective_count < MAX_DEFECTIVE:
            for ext in ['.jpg', '.png', '.JPG', '.PNG']:
                test_path = img_dir / f"{img_stem}_test{ext}"
                if test_path.exists():
                    if not ann_path.exists():
                        continue
                    
                    img_dst = OUT_IMG_DIR / test_path.name
                    ann_dst = OUT_ANN_DIR / test_path.with_suffix(".xml").name
                    
                    success, defect_cnt = create_voc_xml(test_path, ann_path, ann_dst, is_normal=False)
                    
                    if success and defect_cnt > 0:
                        shutil.copy(test_path, img_dst)
                        defective_count += 1
                        
                        if defective_count % 100 == 0:
                            print(f"✓ Defective: {defective_count}/{MAX_DEFECTIVE}")
                    break
        
        # Process NORMAL images (_temp)
        if normal_count < MAX_NORMAL:
            for ext in ['.jpg', '.png', '.JPG', '.PNG']:
                temp_path = img_dir / f"{img_stem}_temp{ext}"
                if temp_path.exists():
                    img_dst = OUT_IMG_DIR / temp_path.name
                    ann_dst = OUT_ANN_DIR / temp_path.with_suffix(".xml").name
                    
                    success, _ = create_voc_xml(temp_path, ann_path, ann_dst, is_normal=True)
                    
                    if success:
                        shutil.copy(temp_path, img_dst)
                        normal_count += 1
                        
                        if normal_count % 100 == 0:
                            print(f"✓ Normal: {normal_count}/{MAX_NORMAL}")
                    break

print("\n" + "="*60)
print(f" Successfully prepared:")
print(f"   Defective images: {defective_count}")
print(f"   Normal images: {normal_count}")
print(f"   Total: {defective_count + normal_count}")
print("="*60)
print(f"\n✓ Images saved to: {OUT_IMG_DIR}")
print(f"✓ Annotations saved to: {OUT_ANN_DIR}")