import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import os
from pathlib import Path


class ICHDataset(Dataset):
    def __init__(self, npy_dir, labels_csv, resize_size=(512, 512)):
        self.npy_dir = Path(npy_dir)
        if not self.npy_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.npy_dir}")
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

        self.labels = pd.read_csv(labels_csv)
        if self.labels.empty:
            raise ValueError(f"{labels_csv} is empty")
        self.image_ids = self.labels['image_id'].values
        self.labels = self.labels['ICH_label'].values

        print(f"Checking {len(self.image_ids)} image_ids against .npy files...")
        self.valid_ids = []
        self.valid_labels = []
        npy_files = set(f.stem for f in self.npy_dir.glob("*.npy"))
        for img_id, label in zip(self.image_ids, self.labels):
            if img_id in npy_files:
                self.valid_ids.append(img_id)
                self.valid_labels.append(label)
            else:
                print(f"Warning: {self.npy_dir / f'{img_id}.npy'} not found, skipping")
        if not self.valid_ids:
            raise ValueError("No valid .npy files match labels.csv image_ids")
        self.image_ids = np.array(self.valid_ids)
        self.labels = np.array(self.valid_labels)
        print(f"Found {len(self.valid_ids)} valid image_id and .npy pairs")

        # Define resize transform
        self.resize = transforms.Resize(resize_size)

        # Verify sample .npy shape
        sample_npy = np.load(self.npy_dir / f"{self.image_ids[0]}.npy")
        print(f"Sample .npy shape: {sample_npy.shape}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.npy_dir / f"{img_id}.npy"
        try:
            img = np.load(img_path).astype(np.float32)
            if img.ndim == 3 and img.shape[0] == 1:
                img = img.squeeze(0)  # [1, H, W] -> [H, W]
            elif img.ndim != 2:
                raise ValueError(f"Unexpected .npy shape for {img_id}: {img.shape}")
            img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
            img = self.resize(img)  # Resize to [1, 512, 512]
            label = self.labels[idx]
            return img, label
        except Exception as e:
            raise FileNotFoundError(f"Failed to load {img_path}: {e}")


def load_resnet50(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = models.resnet50(weights=None)
    # Replace conv1 for 1-channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Identity()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    is_raw_state_dict = 'conv1.weight' in checkpoint
    print(f"Checkpoint type: {'raw state dict' if is_raw_state_dict else 'dict'}")

    if not is_raw_state_dict:
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"Using state dict from key: {key}")
                break
        else:
            raise KeyError("No valid state dict key found in checkpoint (tried: model_state_dict, state_dict, model)")
    else:
        state_dict = checkpoint
        print(f"Raw state dict with {len(state_dict)} keys: {list(state_dict.keys())[:5]}...")
        print(f"conv1.weight shape: {state_dict['conv1.weight'].shape}")

    # Adapt conv1 weights from 3 channels to 1 channel
    if state_dict['conv1.weight'].shape[1] == 3:
        print("Adapting conv1 weights from 3 channels to 1 channel")
        state_dict['conv1.weight'] = state_dict['conv1.weight'].sum(dim=1, keepdim=True)

    # Filter out fc layer weights
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
    print(f"Loading state dict with {len(state_dict)} keys (fc weights excluded)")

    # Load state dict, ignoring mismatches
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    print("Loaded state dict successfully")

    model.eval()
    return model.to('cpu')


def extract_features(model, dataloader, device):
    features, labels = [], []
    with torch.no_grad():
        for batch_idx, (imgs, lbls) in enumerate(dataloader):
            imgs = imgs.to(device)
            feats = model(imgs)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
            if batch_idx % 100 == 0:
                print(f"Processed {batch_idx * dataloader.batch_size} samples")
    return np.concatenate(features), np.concatenate(labels)


def main():
    npy_dir = "" #the file is too large 100GB of size and i cannot upload it with it.
    labels_csv = "labels_filtered.csv"
    checkpoint_path = "models/resnet50_best_100000.pth"
    output_dir = "models/xgboost_v3/"

    print("Verifying paths...")
    if not os.path.exists(npy_dir):
        raise FileNotFoundError(f"npy_dir not found: {npy_dir}")
    npy_count = len(list(Path(npy_dir).glob("*.npy")))
    print(f"Found {npy_count} .npy files in {npy_dir}")
    if npy_count == 0:
        raise ValueError("No .npy files found in npy_dir")
    print("Sample .npy files:", [f.name for f in list(Path(npy_dir).glob("*.npy"))[:5]])

    if not os.path.exists(labels_csv):
        print(f"Warning: {labels_csv} not found, trying labels.csv")
        labels_csv = "F:/processed_100000/labels.csv"
        if not os.path.exists(labels_csv):
            raise FileNotFoundError("Neither labels_filtered.csv nor labels.csv found")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    os.makedirs(output_dir, exist_ok=True)

    device = 'cpu'
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = ICHDataset(npy_dir, labels_csv)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    print("Loading ResNet50...")
    model = load_resnet50(checkpoint_path)

    print("Extracting train features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    print("Extracting val features...")
    val_features, val_labels = extract_features(model, val_loader, device)

    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist',
        device='cpu'
    )
    xgb.fit(train_features, train_labels)

    val_probs = xgb.predict_proba(val_features)[:, 1]
    auc = roc_auc_score(val_labels, val_probs)
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    sensitivity = tpr[thresholds <= 0.2][-1] if any(thresholds <= 0.2) else tpr[0]
    print(f"XGBoost v3 - Val AUC: {auc:.4f}, Val Sensitivity: {sensitivity:.4f} (Threshold: 0.2)")

    xgb.save_model(f"{output_dir}/xgboost_v3.model")
    print(f"Model saved: {output_dir}/xgboost_v3.model")


if __name__ == "__main__":
    main()