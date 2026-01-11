import torch
from torch.utils.data import Dataset
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import torchvision.transforms as T
import numpy as np

CLASS_TO_ID = {
    "open": 1,
    "short": 2,
    "mousebite": 3,
    "spur": 4,
    "copper": 5,
    "pin-hole": 6
}

class PCBDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None, augment=True):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.transforms = transforms
        self.augment = augment
        
        # Color augmentation for robustness
        if self.augment:
            self.color_transform = T.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.02
            )

        # Get all images with annotations
        self.images = []
        for img_path in sorted(self.image_dir.glob("*.jpg")):
            ann_path = self.annotation_dir / img_path.with_suffix(".xml").name
            if ann_path.exists():
                self.images.append(img_path)
        
        for img_path in sorted(self.image_dir.glob("*.png")):
            ann_path = self.annotation_dir / img_path.with_suffix(".xml").name
            if ann_path.exists():
                self.images.append(img_path)
        
        print(f"Found {len(self.images)} images with valid annotations")

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, xml_path):
        """Parse VOC format XML annotation file"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            
            if cls_name not in CLASS_TO_ID:
                continue
            
            bbox = obj.find("bndbox")

            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_ID[cls_name])

        if len(boxes) == 0:
            # Empty - normal PCB
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        ann_path = self.annotation_dir / img_path.with_suffix(".xml").name

        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parse annotations
        boxes, labels = self.parse_voc_xml(ann_path)

        # Apply color augmentation
        if self.augment and len(boxes) > 0:
            # Convert to PIL for augmentation
            from PIL import Image
            pil_image = Image.fromarray(image)
            pil_image = self.color_transform(pil_image)
            image = np.array(pil_image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        # Convert image to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.transforms:
            image = self.transforms(image)

        return image, target