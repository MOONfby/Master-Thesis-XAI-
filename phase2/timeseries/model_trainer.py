"""
InceptionTime model for ECG5000 classification.

InceptionTime (Ismail Fawaz et al. 2020) is a state-of-the-art 1D-CNN for
time series classification, analogous to ResNet-50 for images in Phase 1.

The model is loaded from tsai (PyTorch) or built from scratch if tsai is
unavailable. After training, it is saved to TS_MODEL_PATH.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from phase2.timeseries.config import (
    TS_MODEL_PATH, SERIES_LENGTH, NUM_CLASSES, BATCH_SIZE, LR,
    MAX_EPOCHS, PATIENCE, RANDOM_STATE
)


# ------------------------------------------------------------------
# InceptionTime implementation (self-contained, no tsai dependency)
# ------------------------------------------------------------------

class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, n_filters: int = 32,
                 kernel_sizes=(40, 20, 10)):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, n_filters, kernel_size=1,
                                    bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(n_filters, n_filters, kernel_size=k,
                      padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)
        )
        self.bn = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        bottleneck = self.bottleneck(x)
        parts = [conv(bottleneck) for conv in self.convs]
        parts.append(self.maxpool_conv(x))
        out = torch.cat(parts, dim=1)
        return self.relu(self.bn(out))


class InceptionTime(nn.Module):
    """
    InceptionTime for univariate time series classification.

    Input  : (batch, 1, T)
    Output : (batch, num_classes) — logits
    """
    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES,
                 n_filters: int = 32, depth: int = 6):
        super().__init__()
        n_out = n_filters * 4   # 4 branches per InceptionBlock

        blocks = []
        channels = in_channels
        for i in range(depth):
            blocks.append(InceptionBlock(channels, n_filters))
            channels = n_out
            # Residual shortcut every 3 blocks
            if (i + 1) % 3 == 0:
                blocks.append(_ResidualShortcut(in_channels if i < 3 else n_out,
                                                 n_out))
                in_channels = n_out   # update after first shortcut

        self.inception = nn.Sequential(*blocks)
        self.gap       = nn.AdaptiveAvgPool1d(1)
        self.fc        = nn.Linear(n_out, num_classes)

    def forward(self, x):
        out = self.inception(x)
        out = self.gap(out).squeeze(-1)
        return self.fc(out)


class _ResidualShortcut(nn.Module):
    """Identity or projection shortcut added to InceptionTime residual path."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # x is the accumulated output — shortcut adds projected input
        # This module is used as a no-op skip marker in the Sequential;
        # actual residual addition is handled by InceptionTimeWithResiduals.
        # Here we just pass through.
        return x


class InceptionTimeWithResiduals(nn.Module):
    """
    InceptionTime with explicit residual connections every 3 blocks.
    Standard architecture from the original paper.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES,
                 n_filters: int = 32, depth: int = 6):
        super().__init__()
        n_out = n_filters * 4

        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleDict()

        prev_ch = in_channels
        for i in range(depth):
            self.blocks.append(InceptionBlock(prev_ch, n_filters))
            prev_ch = n_out
            if (i + 1) % 3 == 0:
                block_idx = (i + 1) // 3 - 1
                sc_in = in_channels if block_idx == 0 else n_out
                self.shortcuts[str(block_idx)] = nn.Sequential(
                    nn.Conv1d(sc_in, n_out, 1, bias=False),
                    nn.BatchNorm1d(n_out),
                )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(n_out, num_classes)
        self.relu = nn.ReLU()

        self._in_channels = in_channels
        self._n_out = n_out
        self._depth = depth

    def forward(self, x):
        residual = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i + 1) % 3 == 0:
                sc_key = str((i + 1) // 3 - 1)
                x = self.relu(x + self.shortcuts[sc_key](residual))
                residual = x
        out = self.gap(x).squeeze(-1)
        return self.fc(out)


# ------------------------------------------------------------------
# Training / loading
# ------------------------------------------------------------------

def load_or_train_inceptiontime(data: dict,
                                 force_reload: bool = False) -> nn.Module:
    """
    Load a saved InceptionTime model or train from scratch.

    Parameters
    ----------
    data : dict — output of load_ecg5000()
    force_reload : bool — if True, retrain even if checkpoint exists

    Returns
    -------
    InceptionTimeWithResiduals in eval mode on best available device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = InceptionTimeWithResiduals(
        in_channels=data["X_train"].shape[1],
        num_classes=NUM_CLASSES
    ).to(device)

    TS_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if TS_MODEL_PATH.exists() and not force_reload:
        print(f"  Loading InceptionTime from {TS_MODEL_PATH}")
        state = torch.load(TS_MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model

    print("  Training InceptionTime on ECG5000...")
    _train(model, data, device)
    model.eval()
    return model


def _train(model: nn.Module, data: dict, device: torch.device) -> None:
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    X_tr = torch.tensor(data["X_train"], dtype=torch.float32)
    y_tr = torch.tensor(data["y_train"], dtype=torch.long)

    # 80/20 train/val split
    rng  = np.random.default_rng(RANDOM_STATE)
    N    = len(X_tr)
    idx  = rng.permutation(N)
    n_val = int(0.2 * N)
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = torch.utils.data.TensorDataset(X_tr[train_idx], y_tr[train_idx])
    val_ds   = torch.utils.data.TensorDataset(X_tr[val_idx],   y_tr[val_idx])
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=False
    )

    best_val_loss = float("inf")
    patience_count = 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_idx)

        model.eval()
        val_loss = 0.0
        correct  = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                correct  += (logits.argmax(1) == yb).sum().item()
        val_loss /= len(val_idx)
        val_acc   = correct / len(val_idx)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"    Epoch {epoch:3d}/{MAX_EPOCHS} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_acc={val_acc:.3f} | "
                  f"time={elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), TS_MODEL_PATH)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"    Early stopping at epoch {epoch}")
                break

    # Reload best weights
    state = torch.load(TS_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    print(f"  Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to {TS_MODEL_PATH}")


def make_predict_fn(model: nn.Module) -> callable:
    """
    Return a predict_fn compatible with all explainers and metric functions.

    Parameters
    ----------
    model : InceptionTimeWithResiduals in eval mode

    Returns
    -------
    callable: (np.ndarray (N, 1, T)) -> np.ndarray (N, num_classes)
        Returns softmax probabilities.
    """
    device = next(model.parameters()).device

    def predict_fn(X: np.ndarray) -> np.ndarray:
        model.eval()
        t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(t)
            probs  = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    return predict_fn
