import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json

# =================== CONFIG ===================
MODEL_PATH = "outputs/best_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.50

# 4 main critical defects
CLASS_NAMES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur"
}

CLASS_COLORS = {
    "open": (255, 0, 0),        # Blue
    "short": (0, 255, 0),       # Green
    "mousebite": (0, 0, 255),   # Red
    "spur": (255, 255, 0)       # Cyan
}

def calculate_severity(box):
    """Calculate severity based on defect area"""
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    
    if area < 500:
        return "Low"
    elif area < 1500:
        return "Medium"
    else:
        return "High"
# ==============================================

def load_model(model_path, num_classes=7):
    """Load trained model"""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✓ Model loaded: {model_path}")
    else:
        print(f" Model not found: {model_path}")
        exit(1)
    
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess image for detection"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # To tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original

def detect_defects(model, image_tensor, confidence_threshold=0.85):
    """Run detection and filter only 4 critical defects"""
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        predictions = model(image_tensor)
    
    pred = predictions[0]
    
    # Filter by confidence
    keep = pred['scores'] > confidence_threshold
    boxes = pred['boxes'][keep].cpu().numpy()
    labels = pred['labels'][keep].cpu().numpy()
    scores = pred['scores'][keep].cpu().numpy()
    
    # FILTER: Keep only open, short, mousebite, spur (classes 1, 2, 3, 4)
    critical_defects = []
    for i, label in enumerate(labels):
        if label in [1, 2, 3, 4]:  # Only 4 critical defects
            critical_defects.append(i)
    
    if len(critical_defects) == 0:
        return np.array([]), np.array([]), np.array([])
    
    boxes = boxes[critical_defects]
    labels = labels[critical_defects]
    scores = scores[critical_defects]
    
    return boxes, labels, scores

def analyze_defects(boxes, labels, scores):
    """Output (x,y) centers and severity"""
    results = []
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        
        # Center coordinates
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Severity assessment
        severity = calculate_severity(box)
        
        class_name = CLASS_NAMES.get(label, "unknown")
        
        results.append({
            "defect_type": class_name,
            "confidence": float(score),
            "center_x": center_x,
            "center_y": center_y,
            "bounding_box": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            },
            "severity": severity
        })
    
    return results

def draw_results(image, results, is_normal=False):
    """Draw bounding boxes and labels"""
    output = image.copy()
    h, w = output.shape[:2]
    
    if is_normal:
        # Green border for DEFECT-FREE PCB
        cv2.rectangle(output, (10, 10), (w-10, h-10), (0, 255, 0), 8)
        cv2.putText(output, "DEFECT-FREE PCB", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 6)
        cv2.putText(output, "Quality: PASS", (50, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    else:
        # Red border for defective
        cv2.rectangle(output, (10, 10), (w-10, h-10), (0, 0, 255), 8)
        cv2.putText(output, f"DEFECTIVE - {len(results)} defects", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)
        
        for result in results:
            box = result["bounding_box"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            
            color = CLASS_COLORS.get(result["defect_type"], (0, 255, 0))
            
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            
            # Center point
            cx, cy = result["center_x"], result["center_y"]
            cv2.circle(output, (cx, cy), 6, (0, 0, 255), -1)
            
            # Label
            label_text = f"{result['defect_type']}: {result['confidence']:.0%}"
            severity_text = f"[{result['severity']}]"
            
            cv2.putText(output, label_text, (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(output, severity_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return output

def main():
    print("="*60)
    print("PCB DEFECT DETECTION - 4 CRITICAL DEFECTS")
    print("="*60)
    print("Detecting: open, short, mousebite, spur")
    print("="*60)
    
    # Load model once
    print("\nLoading model...")
    model = load_model(MODEL_PATH)
    
    while True:
        print("\n" + "="*60)
        print("Enter image path (or 'q' to quit):")
        image_path = input("> ").strip()
        
        if image_path.lower() in ['q', 'quit', 'exit']:
            print("\n✓ Exiting...")
            break
        
        if not image_path:
            test_images = list(Path("data/images/train").glob("*_test.jpg"))
            if test_images:
                image_path = test_images[0]
                print(f"Using sample: {image_path}")
            else:
                print(" No sample images found!")
                continue
        
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f" Image not found: {image_path}")
            continue
        
        try:
            print(f"\n Processing: {image_path.name}")
            image_tensor, original = preprocess_image(image_path)
            
            print(" Detecting critical defects...")
            boxes, labels, scores = detect_defects(model, image_tensor, CONFIDENCE_THRESHOLD)
            
            print(" Analyzing...")
            results = analyze_defects(boxes, labels, scores)
            
            # Check if DEFECT-FREE
            if len(results) == 0:
                print(f"\n *** DEFECT-FREE PCB ***")
                print("   Quality Status: PASS")
                print("   No critical defects detected.")
                print("-" * 60)
                
                output_json = Path("outputs") / f"detection_{image_path.stem}.json"
                result_data = {
                    "status": "DEFECT-FREE",
                    "quality": "PASS",
                    "defect_count": 0,
                    "detected_types": ["open", "short", "mousebite", "spur"]
                }
                with open(output_json, "w") as f:
                    json.dump(result_data, f, indent=2)
                print(f"\n✓ Results saved: {output_json}")
                
                result_image = draw_results(original, [], is_normal=True)
                
            else:
                # DEFECTIVE
                print(f"\n  DEFECTIVE PCB - {len(results)} critical defects:")
                print("-" * 60)
                
                for i, result in enumerate(results, 1):
                    print(f"\nDefect {i}:")
                    print(f"  Type:       {result['defect_type']}")
                    print(f"  Confidence: {result['confidence']:.2%}")
                    print(f"  Center:     ({result['center_x']}, {result['center_y']}) pixels")
                    print(f"  Severity:   {result['severity']}")
                
                output_json = Path("outputs") / f"detection_{image_path.stem}.json"
                result_data = {
                    "status": "DEFECTIVE",
                    "quality": "FAIL",
                    "defect_count": len(results),
                    "defects": results,
                    "detected_types": ["open", "short", "mousebite", "spur"]
                }
                with open(output_json, "w") as f:
                    json.dump(result_data, f, indent=2)
                print(f"\n✓ Results saved: {output_json}")
                
                result_image = draw_results(original, results, is_normal=False)
            
            output_img = Path("outputs") / f"detected_{image_path.name}"
            cv2.imwrite(str(output_img), result_image)
            print(f"✓ Image saved: {output_img}")
            
            print("\n" + "="*60)
            print("  IMAGE DISPLAY")
            print("="*60)
            print("Press 'ESC' to continue | Press 'Q' to quit")
            print("="*60)
            
            cv2.imshow("PCB Inspection - ESC: next | Q: quit", result_image)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("\n✓ Next image...")
                    cv2.destroyAllWindows()
                    break
                elif key in [ord('q'), ord('Q')]:
                    print("\n✓ Exiting...")
                    cv2.destroyAllWindows()
                    return
        
        except Exception as e:
            print(f"\n Error: {e}")
            continue
    
    print("\n" + "="*60)
    print(" Thank you!")
    print("="*60)

if __name__ == "__main__":
    main()