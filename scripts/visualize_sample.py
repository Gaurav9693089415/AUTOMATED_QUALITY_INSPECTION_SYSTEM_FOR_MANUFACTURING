import cv2
import xml.etree.ElementTree as ET
from pathlib import Path

# -------- CONFIG --------
IMG_DIR = Path("data/images/train")
ANN_DIR = Path("data/annotations/train")
# ------------------------

# pick first image
img_path = sorted(IMG_DIR.glob("*.jpg"))[0]
ann_path = ANN_DIR / img_path.with_suffix(".xml").name

print(f"Image  : {img_path.name}")
print(f"Annot  : {ann_path.name}")

img = cv2.imread(str(img_path))

tree = ET.parse(ann_path)
root = tree.getroot()

for obj in root.findall("object"):
    cls = obj.find("name").text
    bbox = obj.find("bndbox")

    x1 = int(bbox.find("xmin").text)
    y1 = int(bbox.find("ymin").text)
    x2 = int(bbox.find("xmax").text)
    y2 = int(bbox.find("ymax").text)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img,
        cls,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

cv2.imshow("VOC Annotation Check", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
