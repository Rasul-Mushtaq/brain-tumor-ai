import os
import json
import copy
import kagglehub
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix

# 1. Setup & Data Download
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

# 2. Transforms (Data Augmentation for Training, Clean Normalization for Testing)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

test_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

train_ds = datasets.ImageFolder(os.path.join(path, "Training"), transform=train_tf)
test_ds = datasets.ImageFolder(os.path.join(path, "Testing"), transform=test_tf)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

class_names = train_ds.classes
print("Classes:", class_names)

# 3. Model Initialization
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head with Dropout for regularization
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, len(class_names))
)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return correct / total, all_labels, all_preds

# 4. Stage 1: Train Classifier Head Only (4 Epochs)
print("\nStage 1: Training Classifier Head")
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(4):
    model.train()
    running_loss = 0.0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    acc, _, _ = evaluate(model, test_dl)
    print(f"Epoch {epoch+1}/4 | Loss: {running_loss/len(train_dl):.4f} | Test Acc: {acc*100:.2f}%")

# 5. Stage 2: Unfreeze Layer 3 & Layer 4 for Medical Feature Adaptation (8 Epochs)
print("\nStage 2: Fine-Tuning Convolutional Layers (layer3 & layer4)")
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

best_acc = 0.0
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(8):
    model.train()
    running_loss = 0.0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    
    acc, labels, preds = evaluate(model, test_dl)
    print(f"Epoch {epoch+1}/8 | Loss: {running_loss/len(train_dl):.4f} | Test Accuracy: {acc*100:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        best_weights = copy.deepcopy(model.state_dict())

# 6. Final Evaluation & Export
model.load_state_dict(best_weights)
final_acc, y_true, y_pred = evaluate(model, test_dl)
print(f"\nBest Test Accuracy: {final_acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Save checkpoint with metadata for the FastAPI backend
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "accuracy": final_acc
}, "brain_tumor_resnet18.pth")
print("Saved brain_tumor_resnet18.pth successfully.")