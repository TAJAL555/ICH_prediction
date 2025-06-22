import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import os
import logging

# Setup logging
logging.basicConfig(
    filename='training_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Custom Dataset with Oversampling
class RSNADataset(Dataset):
    def __init__(self, csv_file, npy_folder, transform=None):
        self.df = pd.read_csv(csv_file)
        self.npy_folder = npy_folder
        self.transform = transform
        self.image_ids = sorted(set(self.df['ID'].str.split('_').str[:2].str.join('_')))
        positive_ids = self.df[self.df['Label'] == 1]['ID'].str.split('_').str[:2].str.join('_').unique()
        self.image_ids = list(self.image_ids) + list(positive_ids) * 2
        self.subtypes = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.npy_folder, f'{img_id}.npy')
        img = np.load(img_path).astype(np.float32)
        img = np.stack([img]*3, axis=0)
        labels = self.df[self.df['ID'].str.contains(img_id)][['Label']].values.reshape(-1).astype(np.float32)
        if self.transform:
            img = self.transform(torch.tensor(img))
        return img, torch.tensor(labels)

# Data
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
dataset = RSNADataset(
    csv_file='stage_2_train_sample.csv',
    npy_folder='', #the file is too large 100GB of size and i cannot upload it with it.
    transform=transform
)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=42, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=42, num_workers=0)

# Model
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, 6)
device = torch.device('cpu')
print(f"Training on: {device}")
model = model.to(device)

# Load checkpoint if continuing
checkpoint_path = 'Data/resnet50_best.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded checkpoint: {checkpoint_path}")

# Training
pos_weights = torch.tensor([50.0, 5.0, 10.0, 5.0, 3.0, 1.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1)
num_epochs = 30
best_sensitivity = 0.8594  # From Epoch 1
patience = 2
no_improve = 0

for epoch in range(1, num_epochs):  # Start from Epoch 2
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/10:.4f}')
            logging.info(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/10:.4f}')
            running_loss = 0.0
    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_preds.append(torch.sigmoid(outputs).cpu().numpy())
            val_labels.append(labels.cpu().numpy())
    val_preds = np.vstack(val_preds)
    val_labels = np.vstack(val_labels)
    auc_scores = [roc_auc_score(val_labels[:, i], val_preds[:, i]) for i in range(6)]
    thresholds = [0.2, 0.3, 0.4]
    sensitivity = []
    for t in thresholds:
        sens = [(val_preds[:, i] > t).astype(int)[val_labels[:, i] == 1].mean() for i in range(6)]
        sensitivity.append(np.mean(sens))
    avg_auc = np.mean(auc_scores)
    avg_sensitivity = max(sensitivity)
    best_threshold = thresholds[np.argmax(sensitivity)]
    print(f'Epoch {epoch+1}, Val AUC: {avg_auc:.4f}, Val Sensitivity: {avg_sensitivity:.4f} (Threshold: {best_threshold})')
    logging.info(f'Epoch {epoch+1}, Val AUC: {avg_auc:.4f}, Val Sensitivity: {avg_sensitivity:.4f} (Threshold: {best_threshold})')
    scheduler.step(avg_sensitivity)
    if avg_sensitivity > best_sensitivity:
        best_sensitivity = avg_sensitivity
        no_improve = 0
        torch.save(model.state_dict(), 'Data/resnet50_best_100000.pth')
    else:
        no_improve += 1
    if no_improve >= patience or avg_sensitivity > 0.95:
        print("Early stopping triggered")
        logging.info("Early stopping triggered")
        break
print("Training complete!")
logging.info("Training complete!")