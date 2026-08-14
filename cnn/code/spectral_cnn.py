"""Frozen spectral CNN and deterministic training helpers."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class FrozenSpectralCNN(nn.Module):
    def __init__(self, input_channels: int, classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def loader(x: np.ndarray, y: np.ndarray, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ArrayDataset(x, y), batch_size=64, shuffle=shuffle, num_workers=0,
        generator=generator, pin_memory=torch.cuda.is_available()
    )


@torch.no_grad()
def evaluate_loss(model: nn.Module, data: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    count = 0
    for x, y in data:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        total += float(criterion(model(x), y).item())
        count += int(y.numel())
    return total / count


def fit(
    model: nn.Module,
    train: DataLoader,
    validation: DataLoader,
    device: torch.device,
    maximum_epochs: int = 100,
    patience: int = 12,
) -> tuple[list[dict[str, float]], int]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for x, y in train:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())
            correct += int((logits.argmax(dim=1) == y).sum().item())
        validation_loss = evaluate_loss(model, validation, device)
        history.append(
            {
                "epoch": epoch,
                "train_cross_entropy": total_loss / total,
                "train_accuracy": correct / total,
                "validation_cross_entropy": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("no valid checkpoint")
    model.load_state_dict(best_state)
    return history, best_epoch


@torch.no_grad()
def predict(model: nn.Module, data: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    labels = []
    for x, y in data:
        probabilities.append(torch.softmax(model(x.to(device, non_blocking=True)), dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)
