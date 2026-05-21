from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLPClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dims: list[int],
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    cfg: dict,
    device: torch.device,
    save_path: Path | None = None,
) -> MLPClassifier:
    mcfg = cfg["models"]["mlp"]
    model = MLPClassifier(
        X_train.shape[1],
        n_classes,
        mcfg["hidden_dims"],
        mcfg["dropout"],
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train).long(),
        ),
        batch_size=mcfg["batch_size"],
        shuffle=True,
    )
    val_x = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val).long().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=mcfg["learning_rate"],
        weight_decay=mcfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    # Class weights for imbalance
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = (counts.sum() / (counts + 1e-6)) / len(counts)
    weights = torch.from_numpy(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_f1 = -1.0
    best_state = None
    patience = mcfg["early_stopping_patience"]
    stale = 0

    for epoch in range(mcfg["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            pred = logits.argmax(dim=1).cpu().numpy()
        val_acc = (pred == y_val).mean()
        # macro F1 proxy for early stopping
        from sklearn.metrics import f1_score

        val_f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{mcfg['epochs']}  val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}")
        if stale >= patience:
            print(f"  Early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "n_features": X_train.shape[1],
                "n_classes": n_classes,
                "hidden_dims": mcfg["hidden_dims"],
                "dropout": mcfg["dropout"],
            },
            save_path,
        )
    return model


@torch.no_grad()
def predict_mlp(
    model: MLPClassifier,
    X: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    x = torch.from_numpy(X).to(device)
    logits = model(x)
    proba = torch.softmax(logits, dim=1).cpu().numpy()
    pred = proba.argmax(axis=1)
    return pred, proba
