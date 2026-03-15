"""
Smile Classifier (GENKI-4K) — v2 regularized (PyTorch)

Improved CNN for smile (1) vs non-smile (0) classification on GENKI-4K.
This version builds on the v1 baseline and focuses on reducing overfitting
through train-only augmentation, normalization, batch normalization,
weight decay, learning-rate scheduling, and best-model checkpointing.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms


# ----------------------------
# Config
# ----------------------------

DATA_DIR = os.environ.get("/path/to/GENKI-4K/")
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
SEED = 42
NUM_WORKERS = 4
CHECKPOINT_PATH = "best_model_v2.pth"


# ----------------------------
# Dataset
# ----------------------------
class Genki4KDataset(Dataset):
    """Loads GENKI-4K images and the binary smile label (first column in labels file)."""

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        labels_path = os.path.join(data_dir, "GENKI-4K_Labels.txt")
        images_path = os.path.join(data_dir, "GENKI-4K_Images.txt")
        files_dir = os.path.join(data_dir, "files")

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Missing labels file: {labels_path}")
        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Missing images list file: {images_path}")
        if not os.path.isdir(files_dir):
            raise FileNotFoundError(f"Missing images folder: {files_dir}")

        # Use only the first column: smile/non-smile (0/1)
        self.labels = pd.read_csv(labels_path, sep=r"\s+", header=None, usecols=[0])

        with open(images_path, "r", encoding="utf-8") as f:
            self.image_files = f.read().splitlines()

        if len(self.labels) != len(self.image_files):
            raise ValueError(
                "Mismatch between labels and image list lengths: "
                f"{len(self.labels)} labels vs {len(self.image_files)} image entries."
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_name = f"file{idx + 1:04d}.jpg"
        img_path = os.path.join(self.data_dir, "files", img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image file: {img_path}")

        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels.iloc[idx, 0], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


class ApplyTransform(Dataset):
    """
    Wraps a dataset/subset and applies a transform at access time.
    Useful when train and validation splits need different transforms.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


# ----------------------------
# Model
# ----------------------------
def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )


def build_model() -> nn.Module:
    model = nn.Sequential(
        # Block 1: 64x64 -> 32x32
        conv_block(3, 32),
        # Block 2: 32x32 -> 16x16
        conv_block(32, 64),
        # Block 3: 16x16 -> 8x8
        conv_block(64, 128),
        # Block 4: 8x8 -> 4x4
        conv_block(128, 256),
        # 4x4 -> 1x1 (independent of exact spatial size)
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    return model


# ----------------------------
# Training
# ----------------------------
def train(model: nn.Module, train_dl: DataLoader, valid_dl: DataLoader, device: torch.device):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_correct = 0.0

        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            pred = model(x_batch).squeeze(1)  # shape: (batch,)
            loss = loss_fn(pred, y_batch)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * y_batch.size(0)
            train_correct += ((pred >= 0.5).float() == y_batch).float().sum().item()

        train_loss = train_loss_sum / len(train_dl.dataset)
        train_acc = train_correct / len(train_dl.dataset)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0.0

        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                pred = model(x_batch).squeeze(1)
                loss = loss_fn(pred, y_batch)

                val_loss_sum += loss.item() * y_batch.size(0)
                val_correct += ((pred >= 0.5).float() == y_batch).float().sum().item()

        val_loss = val_loss_sum / len(valid_dl.dataset)
        val_acc = val_correct / len(valid_dl.dataset)

        # Reduce LR if validation accuracy plateaus
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
            f"lr: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Saved new best model to {CHECKPOINT_PATH} (val_acc={best_val_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


def main():
    # Device
    cuda_ok = torch.cuda.is_available()
    print("CUDA available:", cuda_ok)
    if cuda_ok:
        print("GPU:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if cuda_ok else "cpu")
    print("Using device:", device)

    # Reproducibility
    torch.manual_seed(SEED)

    # Transforms
    train_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    valid_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Data
    full_dataset = Genki4KDataset(data_dir=DATA_DIR, transform=None)

    train_size = int(0.85 * len(full_dataset))
    valid_size = len(full_dataset) - train_size

    train_raw, valid_raw = random_split(
        full_dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_dataset = ApplyTransform(train_raw, transform=train_transforms)
    valid_dataset = ApplyTransform(valid_raw, transform=valid_transforms)

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")

    train_dl = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=cuda_ok,
    )
    valid_dl = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=cuda_ok,
    )

    # Model
    model = build_model().to(device)

    # Train
    train(model, train_dl, valid_dl, device)


if __name__ == "__main__":
    main()