from pathlib import Path
from src.dataset import PCBDetectionDataset
import cv2
import xml.etree.ElementTree as ET

print("="*60)
print("TESTING PREPARED DATASET")
print("="*60)

# Initialize dataset
ds = PCBDetectionDataset(
    "data/images/train",
    "data/annotations/train"
)

print(f"\n✓ Total images in dataset: {len(ds)}")

if len(ds) == 0:
    print("❌ No images found!")
    exit(1)

# Test first few samples
print("\n" + "="*60)
print("SAMPLE DATA CHECK")
print("="*60)

for i in range(min(3, len(ds))):
    img, target = ds[i]
    print(f"\nSample {i+1}:")
    print(f"  Image shape: {img.shape}")
    print(f"  Number of defects: {len(target['boxes'])}")
    print(f"  Defect classes: {target['labels'].tolist()}")
    print(f"  Bounding boxes: {target['boxes'].shape}")

# Visualize first image
print("\n" + "="*60)
print("VISUALIZING FIRST IMAGE")
print("="*60)

img_path = sorted(Path("data/images/train").glob("*.jpg"))[0]
ann_path = Path("data/annotations/train") / img_path.with_suffix(".xml").name

print(f"Image: {img_path.name}")
print(f"Annotation: {ann_path.name}")

img = cv2.imread(str(img_path))

# Parse XML
tree = ET.parse(ann_path)
root = tree.getroot()

CLASS_COLORS = {
    "open": (255, 0, 0),        # Blue
    "short": (0, 255, 0),       # Green
    "mousebite": (0, 0, 255),   # Red
    "spur": (255, 255, 0),      # Cyan
    "copper": (255, 0, 255),    # Magenta
    "pin-hole": (0, 255, 255)   # Yellow
}

defect_count = 0
for obj in root.findall("object"):
    cls = obj.find("name").text
    bbox = obj.find("bndbox")

    x1 = int(bbox.find("xmin").text)
    y1 = int(bbox.find("ymin").text)
    x2 = int(bbox.find("xmax").text)
    y2 = int(bbox.find("ymax").text)

    color = CLASS_COLORS.get(cls, (0, 255, 0))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        cls,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )
    defect_count += 1

print(f"\n✓ Found {defect_count} defects in first image")
print("\nDisplaying image... (Press any key to close)")

cv2.imshow("PCB Defect Detection - Sample", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n" + "="*60)
print("✅ DATASET TEST COMPLETE!")
print("="*60)
print("\nYour dataset is ready for training!")