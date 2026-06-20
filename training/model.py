import numpy as np

MIN_POSITIVE_COUNT_FOR_OUTPUT = 50
MAX_POS_WEIGHT = 8.0


def require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        raise SystemExit(1)


def _split_and_normalize(X, y, test_split):
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split_index = int(n_samples * (1 - test_split))
    train_idx, test_idx = indices[:split_index], indices[split_index:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)

    return X_train, y_train, X_test, y_test, mean, std


def _resolve_settings(settings):
    s = dict(settings or {})
    s.setdefault("batch_size", 32)
    s.setdefault("epochs", 500)
    s.setdefault("learning_rate", 0.001)
    s.setdefault("test_split", 0.2)
    s.setdefault("hidden1", 32)
    s.setdefault("hidden2", 16)
    s.setdefault("min_positive_count_for_output", MIN_POSITIVE_COUNT_FOR_OUTPUT)
    s.setdefault("max_pos_weight", MAX_POS_WEIGHT)
    s.setdefault("positive_class_weight_boost", 1.0)
    s.setdefault("mask_logit_threshold", 0.0)
    s.setdefault("log_every_epochs", 20)
    return s


def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def _evaluate_metrics(model, data_loader, criterion_flip, active_mask_t, threshold):
    import torch

    model.eval()
    active_mask_bool = active_mask_t > 0.5

    val_loss_sum = 0.0
    bit_correct = 0
    bit_total = 0
    exact = 0
    total = 0
    tp = 0
    fp = 0
    fn = 0
    pred_pos = 0
    true_pos = 0

    with torch.no_grad():
        for xb, yb in data_loader:
            flip_logits = model(xb)
            loss_flip_all = criterion_flip(flip_logits, yb)
            active_count = active_mask_t.sum().clamp_min(1.0)
            loss = (loss_flip_all * active_mask_t).sum() / (yb.shape[0] * active_count)
            val_loss_sum += loss.item() * len(xb)

            pred = (flip_logits > float(threshold)).float()
            pred_b = pred.bool()
            yb_b = yb.bool()

            active_cols = active_mask_bool.unsqueeze(0).expand_as(pred_b)
            pred_a = pred_b[active_cols]
            y_a = yb_b[active_cols]

            bit_correct += (pred_a == y_a).sum().item()
            bit_total += pred_a.numel()

            eq_active = (pred_b == yb_b) | (~active_cols)
            exact += eq_active.all(dim=1).sum().item()
            total += yb.shape[0]

            tp += (pred_a & y_a).sum().item()
            fp += (pred_a & (~y_a)).sum().item()
            fn += ((~pred_a) & y_a).sum().item()
            pred_pos += pred_a.sum().item()
            true_pos += y_a.sum().item()

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    bit_acc = _safe_div(bit_correct, bit_total)
    exact_acc = _safe_div(exact, total)

    return {
        "val_loss": val_loss_sum,
        "bit_acc": bit_acc,
        "exact_acc": exact_acc,
        "hamming_loss": 1.0 - bit_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_positive_rate": _safe_div(pred_pos, bit_total),
        "true_positive_rate": _safe_div(true_pos, bit_total),
        "samples": int(total),
        "active_bits_per_sample": int(active_mask_bool.sum().item()),
    }


def train_pytorch_mask_only(X, y, settings=None):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    settings = _resolve_settings(settings)

    class EscapeNetMaskOnly(nn.Module):
        def __init__(self, in_dim, out_bits):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, settings["hidden1"])
            self.fc2 = nn.Linear(settings["hidden1"], settings["hidden2"])
            self.flip_head = nn.Linear(settings["hidden2"], out_bits)

        def forward(self, x):
            h = torch.relu(self.fc1(x))
            h = torch.relu(self.fc2(h))
            return self.flip_head(h)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    output_bits = y.shape[1]
    X_train, y_train, X_test, y_test, mean, std = _split_and_normalize(
        X, y, settings["test_split"]
    )

    train_ds = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test),
    )
    model = EscapeNetMaskOnly(X.shape[1], output_bits)

    train_dl = DataLoader(train_ds, batch_size=settings["batch_size"], shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=settings["batch_size"])

    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])

    pos_counts = y_train.sum(axis=0).astype(np.float64)
    neg_counts = y_train.shape[0] - pos_counts
    active_outputs = pos_counts >= settings["min_positive_count_for_output"]
    safe_pos_counts = np.maximum(pos_counts, 1.0)
    raw_pos_weight = neg_counts / safe_pos_counts
    pos_weight = np.sqrt(raw_pos_weight) * 0.9
    pos_weight = pos_weight * float(settings["positive_class_weight_boost"])
    pos_weight = np.clip(pos_weight, 0.5, settings["max_pos_weight"]).astype(np.float32)

    pos_weight_t = torch.tensor(pos_weight)
    active_mask_t = torch.tensor(active_outputs.astype(np.float32))

    criterion_flip = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t, reduction="none")

    best_exact = 0.0
    best_state = None
    best_epoch = -1
    best_metrics = None
    history = []

    for epoch in range(settings["epochs"]):
        model.train()
        train_loss = 0.0

        for xb, yb in train_dl:
            flip_logits = model(xb)
            loss_flip_all = criterion_flip(flip_logits, yb)
            active_count = active_mask_t.sum().clamp_min(1.0)
            loss = (loss_flip_all * active_mask_t).sum() / (yb.shape[0] * active_count)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(xb)

        model.eval()
        eval_m = _evaluate_metrics(
            model,
            test_dl,
            criterion_flip,
            active_mask_t,
            settings["mask_logit_threshold"],
        )
        exact_acc = eval_m["exact_acc"]
        bit_acc = eval_m["bit_acc"]

        history_row = {
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss / len(train_ds)),
            "val_loss": float(eval_m["val_loss"] / len(test_ds)) if len(test_ds) > 0 else 0.0,
            "exact_acc": float(exact_acc),
            "bit_acc": float(bit_acc),
            "hamming_loss": float(eval_m["hamming_loss"]),
            "precision": float(eval_m["precision"]),
            "recall": float(eval_m["recall"]),
            "f1": float(eval_m["f1"]),
            "pred_positive_rate": float(eval_m["pred_positive_rate"]),
            "true_positive_rate": float(eval_m["true_positive_rate"]),
        }
        history.append(history_row)

        if exact_acc > best_exact:
            best_exact = exact_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            best_metrics = dict(history_row)

        log_every = int(settings.get("log_every_epochs", 20))
        if log_every <= 0:
            log_every = 20
        if (epoch + 1) % log_every == 0:
            print(
                f"[mask_only] Epoch {epoch+1:3d} | "
                f"TrainLoss {history_row['train_loss']:.4f} | "
                f"ValLoss {history_row['val_loss']:.4f} | "
                f"Bit {history_row['bit_acc']:.3f} | "
                f"Exact {history_row['exact_acc']:.3f} | "
                f"F1 {history_row['f1']:.3f} | "
                f"Prec {history_row['precision']:.3f} | "
                f"Rec {history_row['recall']:.3f}"
            )

    if best_state:
        model.load_state_dict(best_state)

    print(f"\n[mask_only] Best exact: {best_exact:.4f} (epoch {best_epoch})")
    if best_metrics is not None:
        print(
            "[mask_only] Best metrics | "
            f"ValLoss {best_metrics['val_loss']:.4f} | "
            f"Bit {best_metrics['bit_acc']:.3f} | "
            f"Exact {best_metrics['exact_acc']:.3f} | "
            f"F1 {best_metrics['f1']:.3f} | "
            f"Prec {best_metrics['precision']:.3f} | "
            f"Rec {best_metrics['recall']:.3f}"
        )

    metadata = {
        "mask_logit_threshold": float(settings["mask_logit_threshold"]),
        "active_outputs": int(active_outputs.sum()),
        "total_outputs": int(output_bits),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "history": history,
    }

    return model, mean, std, output_bits, metadata


def extract_weights_pytorch(model):
    state = model.state_dict()

    return {
        "w1": state["fc1.weight"].detach().cpu().numpy(),
        "b1": state["fc1.bias"].detach().cpu().numpy(),
        "w2": state["fc2.weight"].detach().cpu().numpy(),
        "b2": state["fc2.bias"].detach().cpu().numpy(),
        "w_flip": state["flip_head.weight"].detach().cpu().numpy(),
        "b_flip": state["flip_head.bias"].detach().cpu().numpy(),
    }
