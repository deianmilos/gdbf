import numpy as np

from training_config import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN1,
    HIDDEN2,
    LEARNING_RATE,
    OUTPUT_BITS,
    TEST_SPLIT,
    THRESHOLD,
)


def require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        raise SystemExit(1)


def train_pytorch(X, y):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split_index = int(n_samples * (1 - TEST_SPLIT))
    train_idx, test_idx = indices[:split_index], indices[split_index:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-6] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = nn.Sequential(
        nn.Linear(X.shape[1], HIDDEN1),
        nn.ReLU(),
        nn.Linear(HIDDEN1, HIDDEN2),
        nn.ReLU(),
        nn.Linear(HIDDEN2, OUTPUT_BITS),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    y_train_t = torch.tensor(y_train)
    pos_rate = y_train_t.mean(dim=0).clamp(1e-3, 1 - 1e-3)
    pos_weight = (1.0 - pos_rate) / pos_rate
    print(f"Positive rates (train): {pos_rate.tolist()}")
    print(f"Pos weights applied:    {pos_weight.tolist()}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.unsqueeze(0))

    best_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for xb, yb in train_dl:
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in test_dl:
                logits = model(xb)
                pred_binary = (torch.sigmoid(logits) > THRESHOLD).float()
                correct += (pred_binary == yb).sum().item()
                total += yb.numel()

        acc = correct / total if total > 0 else 0.0
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            avg_loss = train_loss / len(train_ds)
            print(f"Epoch {epoch + 1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    print(f"\nBest test accuracy: {best_acc:.4f}")
    _print_bit_metrics(model, X_test, y_test)

    return model, mean, std


def _print_bit_metrics(model, X_test, y_test):
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test))
        pred_binary = (torch.sigmoid(logits) > THRESHOLD).float()
        y_test_t = torch.tensor(y_test)

        for bit in range(OUTPUT_BITS):
            tp = ((pred_binary[:, bit] == 1) & (y_test_t[:, bit] == 1)).sum().item()
            fp = ((pred_binary[:, bit] == 1) & (y_test_t[:, bit] == 0)).sum().item()
            fn = ((pred_binary[:, bit] == 0) & (y_test_t[:, bit] == 1)).sum().item()
            tn = ((pred_binary[:, bit] == 0) & (y_test_t[:, bit] == 0)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            print(
                f"  Bit {bit}: Precision={precision:.3f} Recall={recall:.3f} "
                f"TP={tp} FP={fp} FN={fn} TN={tn}"
            )


def extract_weights_pytorch(model):
    state = model.state_dict()
    w1 = state["0.weight"].numpy()
    b1 = state["0.bias"].numpy()
    w2 = state["2.weight"].numpy()
    b2 = state["2.bias"].numpy()
    w3 = state["4.weight"].numpy()
    b3 = state["4.bias"].numpy()
    return w1, b1, w2, b2, w3, b3
