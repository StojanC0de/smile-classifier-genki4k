"""
Smile Classifier (GENKI-4K) — v1 baseline (PyTorch)

Baseline CNN for smile (1) vs non-smile (0). This version is intentionally simple and is
meant to establish a working training/validation pipeline (often shows a generalization gap).
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
# Set this environment variable instead of hardcoding paths:
#   export GENKI4K_DIR="/path/to/GENKI-4K"
DATA_DIR = os.environ.get("GENKI4K_DIR", "/path/to/GENKI-4K/")
BATCH_SIZE = 32
NUM_EPOCHS = 300
SEED = 1
NUM_WORKERS = 4


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

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Missing labels file: {labels_path}")
        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Missing images list file: {images_path}")

        # Use only the first column: smile/non-smile (0/1)
        self.labels = pd.read_csv(labels_path, sep=r"\s+", header=None, usecols=[0])

        with open(images_path, "r", encoding="utf-8") as f:
            self.image_files = f.read().splitlines()

        files_dir = os.path.join(data_dir, "files")
        if not os.path.isdir(files_dir):
            raise FileNotFoundError(f"Missing images folder: {files_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        # GENKI images are typically named file0001.jpg, file0002.jpg, ...
        img_name = f"file{idx + 1:04d}.jpg"
        img_path = os.path.join(self.data_dir, "files", img_name)

        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels.iloc[idx, 0], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


# ----------------------------
# Model
# ----------------------------
def build_model() -> nn.Module:
    model = nn.Sequential(
        # Block 1
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.5),
        # Block 2
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.5),
        # Block 3
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        # Block 4
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=8),  # 8x8 -> 1x1
        nn.Flatten(),
        nn.Linear(256, 1),
        nn.Sigmoid(),  # v1 uses Sigmoid + BCELoss
    )
    return model


# ----------------------------
# Training
# ----------------------------
def train(model: nn.Module, train_dl: DataLoader, valid_dl: DataLoader, device: torch.device):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_correct = 0.0

        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            pred = model(x_batch)[:, 0]  # shape: (batch,)
            loss = loss_fn(pred, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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

                pred = model(x_batch)[:, 0]
                loss = loss_fn(pred, y_batch)

                val_loss_sum += loss.item() * y_batch.size(0)
                val_correct += ((pred >= 0.5).float() == y_batch).float().sum().item()

        val_loss = val_loss_sum / len(valid_dl.dataset)
        val_acc = val_correct / len(valid_dl.dataset)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
        )


def main():
    # Device
    cuda_ok = torch.cuda.is_available()
    print("CUDA available:", cuda_ok)
    if cuda_ok:
        print("GPU:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if cuda_ok else "cpu")
    print("Using device:", device)

    # Reproducibility (baseline)
    torch.manual_seed(SEED)

    # Transforms (v1: simple resize + tensor)
    image_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    # Data
    dataset = Genki4KDataset(data_dir=DATA_DIR, transform=image_transforms)

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

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