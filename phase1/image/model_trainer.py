"""
ResNet-50 fine-tuning on CUB-200-2011 for Phase 1 Image XAI Evaluation.

Uses transfer learning: ImageNet-pretrained ResNet-50 with replaced
final fully-connected layer, fine-tuned for 200-class bird classification.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50, ResNet50_Weights

from phase1.image.config import (
    IMAGE_MODEL_PATH, NUM_CLASSES, RANDOM_STATE
)

# Fine-tuning hyperparameters
_EPOCHS      = 30
_BATCH_SIZE  = 64
_LR          = 1e-4
_WEIGHT_DECAY = 1e-4


def load_or_finetune_resnet50(data: dict,
                               force_retrain: bool = False) -> nn.Module:
    """
    Load saved ResNet-50 checkpoint or fine-tune from scratch.

    Parameters
    ----------
    data : dict
        Output of load_cub200() — must contain X_train, y_train, X_test, y_test.
    force_retrain : bool
        If True, always retrain even if checkpoint exists.

    Returns
    -------
    nn.Module
        ResNet-50 in eval mode on the best available device.
    """
    IMAGE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_model().to(device)

    if IMAGE_MODEL_PATH.exists() and not force_retrain:
        print(f"  Loading ResNet-50 checkpoint from {IMAGE_MODEL_PATH}")
        state = torch.load(IMAGE_MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model

    print(f"  Fine-tuning ResNet-50 on CUB-200-2011 (device={device})...")
    _finetune(model, data, device)
    torch.save(model.state_dict(), IMAGE_MODEL_PATH)
    print(f"  Checkpoint saved to {IMAGE_MODEL_PATH}")
    model.eval()
    return model


def get_predict_fn(model: nn.Module):
    """
    Return a callable predict_fn(images) -> np.ndarray (N, num_classes).

    Compatible with the metric functions (same pattern as Phase 1 tabular
    model.predict_proba).

    Parameters
    ----------
    model : nn.Module (eval mode)

    Returns
    -------
    callable : (np.ndarray (N,3,H,W) float32) -> np.ndarray (N, NUM_CLASSES)
    """
    device = next(model.parameters()).device

    def predict_fn(images: np.ndarray) -> np.ndarray:
        if images.ndim == 3:
            images = images[None]
        t = torch.tensor(images, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(t)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    return predict_fn


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _build_model() -> nn.Module:
    torch.manual_seed(RANDOM_STATE)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model


def _finetune(model: nn.Module, data: dict, device: torch.device):
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.long)
    X_test  = torch.tensor(data["X_test"],  dtype=torch.float32)
    y_test  = torch.tensor(data["y_test"],  dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=_BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=_BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    # Fine-tune all layers but use higher LR for the new FC head
    optimizer = optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "fc" not in n],
         "lr": _LR * 0.1},
        {"params": model.fc.parameters(), "lr": _LR},
    ], weight_decay=_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_EPOCHS)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, _EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

        scheduler.step()
        train_acc = correct / total
        val_acc   = _eval_accuracy(model, test_loader, device)
        elapsed   = time.time() - t0

        print(f"    Epoch {epoch:2d}/{_EPOCHS} | "
              f"loss={total_loss/total:.4f} | "
              f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | "
              f"{elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"  Best val accuracy: {best_acc:.4f}")
    model.load_state_dict(best_state)
    model.eval()


def _eval_accuracy(model: nn.Module, loader: DataLoader,
                   device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(1)
            correct += (preds == y_batch).sum().item()
            total   += len(y_batch)
    return correct / total
