import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import timm  # Library for advanced models

# === Check and setup CUDA with mixed precision ===
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please run on a machine with GPU.")
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True  # Optimizes CUDA operations
scaler = GradScaler()  # For mixed precision training
print(f"Using device: {device}")

# === Data paths ===
base_dir = "C:/Users/noore/OneDrive/Desktop/burn skinnn/skin burn dataset"
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")

# === Enhanced Transforms ===
# Medical images need careful augmentation - preserving diagnostic features
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=(3, 3)),  # Mild blur to simulate focus variations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Imagenet stats
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Memory-efficient Data Loading ===
# For large datasets, we need to be careful with memory
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)

# Use pin_memory=True for faster GPU transfer and drop_last to avoid partial batches
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, 
                         num_workers=4, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False,
                         num_workers=4, pin_memory=True)

# === Advanced Model Architecture ===
class BurnClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BurnClassifier, self).__init__()
        # Using EfficientNet - lightweight and effective
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, features_only=False)
        
        # Freeze early layers
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze last few layers
        for param in self.backbone.blocks[-3:].parameters():
            param.requires_grad = True
            
        # Replace classifier head
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Increased dropout for regularization
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# === Model, Loss, Optimizer ===
model = BurnClassifier(num_classes=len(train_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Helps with class imbalance
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # AdamW better for generalization

# Learning rate scheduler
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                                        steps_per_epoch=len(train_loader),
                                        epochs=20,
                                        pct_start=0.3)

# Early stopping
best_val_acc = 0.0
patience = 3
no_improve = 0

# === Enhanced Training Loop ===
num_epochs = 20
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # --- Training with mixed precision ---
    model.train()
    running_loss, correct = 0.0, 0
    train_bar = tqdm(train_loader, desc="Training")
    
    for images, labels in train_bar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():  # Mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        
        train_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / len(train_loader.dataset)
    print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # --- Validation ---
    model.eval()
    val_loss, val_correct = 0.0, 0
    val_bar = tqdm(valid_loader, desc="Validating")
    
    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            
            val_bar.set_postfix(loss=loss.item())
    
    val_loss /= len(valid_loader.dataset)
    val_acc = val_correct / len(valid_loader.dataset)
    print(f"Valid Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    # Early stopping check
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve = 0
        torch.save(model.state_dict(), "best_burn_model.pth")
        print("Model improved - saved new best model")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"No improvement for {patience} epochs, stopping early")
            break

# === Save Final Model ===
torch.save(model.state_dict(), "final_burn_model.pth")
print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")