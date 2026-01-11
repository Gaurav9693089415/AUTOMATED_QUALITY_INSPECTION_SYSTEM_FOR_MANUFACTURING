import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from pathlib import Path
import time
import sys
sys.path.append(str(Path(__file__).parent))
from src.dataset import PCBDetectionDataset

# =================== CONFIG ===================
NUM_CLASSES = 7  # 6 defects + background
BATCH_SIZE = 2
NUM_EPOCHS = 12
LEARNING_RATE = 0.005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Path("outputs").mkdir(exist_ok=True)

print("="*60)
print("PCB DEFECT DETECTION - COMPLETE TRAINING")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print("="*60)
# ==============================================

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Load dataset
print("\nLoading dataset...")
dataset = PCBDetectionDataset(
    "data/images/train",
    "data/annotations/train",
    augment=True  # Color augmentation enabled
)

# Split train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

# Model
print("\nInitializing model...")
model = get_model(NUM_CLASSES)
model.to(DEVICE)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=0.0005
)

# Training
print("\n" + "="*60)
print("TRAINING START")
print("="*60)

best_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
    start_time = time.time()
    
    for i, (images, targets) in enumerate(train_loader):
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_loss += losses.item()
        batch_count += 1
        
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(train_loader)}] Loss: {losses.item():.4f}")
    
    avg_loss = epoch_loss / batch_count
    
    # Validation
    model.train()
    val_loss = 0
    val_count = 0
    for images, targets in val_loader:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
            val_count += 1
    
    avg_val_loss = val_loss / val_count if val_count > 0 else 0
    epoch_time = time.time() - start_time
    
    print(f"\n  Train Loss: {avg_loss:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    print(f"  Time: {epoch_time:.1f}s")
    
    # Save best model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), "outputs/best_model.pth")
        print(f"  ✓ Best model saved!")
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"outputs/checkpoint_epoch_{epoch+1}.pth")
        print(f"  ✓ Checkpoint saved")

# Save final
torch.save(model.state_dict(), "outputs/final_model.pth")

print("\n" + "="*60)
print(" TRAINING COMPLETE!")
print("="*60)
print(f"Best val loss: {best_loss:.4f}")
print(f"Models saved in: outputs/")