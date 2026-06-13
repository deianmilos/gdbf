import numpy as np

from config import (
    BATCH_SIZE,
    BUDGET_LOSS_WEIGHT,
    EPOCHS,
    HIDDEN1,
    HIDDEN2,
    LEARNING_RATE,
    TEST_SPLIT,
)

# Default budget setup for budgeted training variant.
VALID_BUDGETS = np.array([0, 2, 4], dtype=np.int64)
FILTER_INVALID_BUDGET_ROWS = True
MIN_POSITIVE_COUNT_FOR_OUTPUT = 50
MAX_POS_WEIGHT = 8.0


def require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        raise SystemExit(1)


def _mask_weight_to_budget_class(weights, valid_budgets):
    classes = np.zeros_like(weights, dtype=np.int64)
    for i, w in enumerate(weights):
        nearest = int(np.argmin(np.abs(valid_budgets - int(w))))
        classes[i] = nearest
    return classes


def _predict_mask_with_budget(flip_logits, budget_logits, valid_budgets):
    import torch

    vb = torch.tensor(valid_budgets, device=flip_logits.device, dtype=torch.long)
    batch_size, out_bits = flip_logits.shape

    budget_class = torch.argmax(budget_logits, dim=1)
    budget_weight = vb[budget_class]

    pred = torch.zeros_like(flip_logits)

    for i in range(batch_size):
        k = int(budget_weight[i].item())
        if k <= 0:
            continue
        if k > out_bits:
            k = out_bits

        top_idx = torch.topk(flip_logits[i], k=k).indices
        pred[i, top_idx] = 1.0

    return pred, budget_class, budget_weight


def _filter_invalid_budget_rows(X, y, valid_budgets):
    weights = np.rint(np.sum(y, axis=1)).astype(np.int64)
    unique, counts = np.unique(weights, return_counts=True)
    print("Mask weight distribution before filtering:")
    for u, c in zip(unique, counts):
        print(f"  weight {int(u)}: {int(c)} rows")

    valid = np.isin(weights, valid_budgets)
    if not np.all(valid):
        removed = int(np.sum(~valid))
        print(f"Filtering invalid budget rows: removed {removed} rows")

    Xf = X[valid]
    yf = y[valid]

    weights_after = np.rint(np.sum(yf, axis=1)).astype(np.int64)
    unique_after, counts_after = np.unique(weights_after, return_counts=True)
    print("Mask weight distribution after filtering:")
    for u, c in zip(unique_after, counts_after):
        print(f"  weight {int(u)}: {int(c)} rows")

    return Xf, yf


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
    s.setdefault("batch_size", BATCH_SIZE)
    s.setdefault("epochs", EPOCHS)
    s.setdefault("learning_rate", LEARNING_RATE)
    s.setdefault("test_split", TEST_SPLIT)
    s.setdefault("hidden1", HIDDEN1)
    s.setdefault("hidden2", HIDDEN2)
    s.setdefault("budget_loss_weight", BUDGET_LOSS_WEIGHT)
    s.setdefault("valid_budgets", VALID_BUDGETS.tolist())
    s.setdefault("filter_invalid_budget_rows", FILTER_INVALID_BUDGET_ROWS)
    s.setdefault("min_positive_count_for_output", MIN_POSITIVE_COUNT_FOR_OUTPUT)
    s.setdefault("max_pos_weight", MAX_POS_WEIGHT)
    s.setdefault("positive_class_weight_boost", 1.0)
    s.setdefault("mask_logit_threshold", 0.0)
    return s


def train_pytorch_variant(X, y, variant="budgeted", settings=None):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    settings = _resolve_settings(settings)

    class EscapeNetBudgeted(nn.Module):
        def __init__(self, in_dim, out_bits, budget_classes):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, settings["hidden1"])
            self.fc2 = nn.Linear(settings["hidden1"], settings["hidden2"])
            self.flip_head = nn.Linear(settings["hidden2"], out_bits)
            self.budget_head = nn.Linear(settings["hidden2"], budget_classes)

        def forward(self, x):
            h = torch.relu(self.fc1(x))
            h = torch.relu(self.fc2(h))
            return self.flip_head(h), self.budget_head(h)

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

    valid_budgets = np.asarray(settings["valid_budgets"], dtype=np.int64)

    if variant == "budgeted" and settings["filter_invalid_budget_rows"]:
        X, y = _filter_invalid_budget_rows(X, y, valid_budgets)

    output_bits = y.shape[1]
    X_train, y_train, X_test, y_test, mean, std = _split_and_normalize(
        X, y, settings["test_split"]
    )

    if variant == "budgeted":
        budget_train = _mask_weight_to_budget_class(
            np.rint(np.sum(y_train, axis=1)).astype(np.int64),
            valid_budgets,
        )
        budget_test = _mask_weight_to_budget_class(
            np.rint(np.sum(y_test, axis=1)).astype(np.int64),
            valid_budgets,
        )

        train_ds = TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train),
            torch.tensor(budget_train),
        )
        test_ds = TensorDataset(
            torch.tensor(X_test),
            torch.tensor(y_test),
            torch.tensor(budget_test),
        )

        model = EscapeNetBudgeted(X.shape[1], output_bits, len(valid_budgets))
    else:
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

    if variant == "budgeted":
        budget_counts = np.bincount(
            np.rint(np.sum(y_train, axis=1)).astype(np.int64),
            minlength=max(1, int(valid_budgets.max()) + 1),
        ).astype(np.float64)
        class_counts = np.bincount(
            _mask_weight_to_budget_class(np.rint(np.sum(y_train, axis=1)).astype(np.int64), valid_budgets),
            minlength=len(valid_budgets),
        ).astype(np.float64)
        class_counts[class_counts < 1] = 1
        budget_weights = class_counts.sum() / (len(valid_budgets) * class_counts)
        budget_weights = np.clip(budget_weights, 1.0, 4.0)
        criterion_budget = nn.CrossEntropyLoss(weight=torch.tensor(budget_weights, dtype=torch.float32))

        print(f"Budget counts: {budget_counts.tolist()}")
        print(f"Budget class weights: {budget_weights.tolist()}")

    best_exact = 0.0
    best_aux = 0.0
    best_state = None

    for epoch in range(settings["epochs"]):
        model.train()
        train_loss = 0.0

        for batch in train_dl:
            if variant == "budgeted":
                xb, yb, kb = batch
                flip_logits, budget_logits = model(xb)
                loss_flip_all = criterion_flip(flip_logits, yb)
                active_count = active_mask_t.sum().clamp_min(1.0)
                loss_flip = (loss_flip_all * active_mask_t).sum() / (yb.shape[0] * active_count)
                loss_budget = criterion_budget(budget_logits, kb)
                loss = loss_flip + (1.5 * settings["budget_loss_weight"]) * loss_budget
            else:
                xb, yb = batch
                flip_logits = model(xb)
                loss_flip_all = criterion_flip(flip_logits, yb)
                active_count = active_mask_t.sum().clamp_min(1.0)
                loss = (loss_flip_all * active_mask_t).sum() / (yb.shape[0] * active_count)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(xb)

        model.eval()
        exact = 0
        bit_correct = 0
        bit_total = 0
        aux_correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_dl:
                if variant == "budgeted":
                    xb, yb, kb = batch
                    flip_logits, budget_logits = model(xb)
                    pred, budget_class_pred, _ = _predict_mask_with_budget(
                        flip_logits,
                        budget_logits,
                        valid_budgets,
                    )
                    aux_correct += (budget_class_pred == kb).sum().item()
                else:
                    xb, yb = batch
                    flip_logits = model(xb)
                    pred = (flip_logits > float(settings["mask_logit_threshold"])).float()

                exact += (pred == yb).all(dim=1).sum().item()
                bit_correct += (pred == yb).sum().item()
                bit_total += yb.numel()
                total += yb.shape[0]

        exact_acc = exact / total if total > 0 else 0.0
        bit_acc = bit_correct / bit_total if bit_total > 0 else 0.0
        aux_acc = aux_correct / total if (total > 0 and variant == "budgeted") else 0.0

        if exact_acc > best_exact or (exact_acc == best_exact and aux_acc > best_aux):
            best_exact = exact_acc
            best_aux = aux_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            if variant == "budgeted":
                print(
                    f"[{variant}] Epoch {epoch+1:3d} | Loss {train_loss/len(train_ds):.4f} | "
                    f"Bit {bit_acc:.3f} | Exact {exact_acc:.3f} | Budget {aux_acc:.3f}"
                )
            else:
                print(
                    f"[{variant}] Epoch {epoch+1:3d} | Loss {train_loss/len(train_ds):.4f} | "
                    f"Bit {bit_acc:.3f} | Exact {exact_acc:.3f}"
                )

    if best_state:
        model.load_state_dict(best_state)

    if variant == "budgeted":
        print(f"\n[{variant}] Best exact: {best_exact:.4f}, Best budget: {best_aux:.4f}")
    else:
        print(f"\n[{variant}] Best exact: {best_exact:.4f}")

    metadata = {
        "variant": variant,
        "valid_budgets": valid_budgets.tolist(),
        "mask_logit_threshold": float(settings["mask_logit_threshold"]),
    }

    return model, mean, std, output_bits, metadata


def train_pytorch(X, y):
    # Backward-compatible default path
    model, mean, std, output_bits, _ = train_pytorch_variant(X, y, variant="budgeted", settings=None)
    return model, mean, std, output_bits


def extract_weights_pytorch(model):
    state = model.state_dict()

    out = {
        "w1": state["fc1.weight"].detach().cpu().numpy(),
        "b1": state["fc1.bias"].detach().cpu().numpy(),
        "w2": state["fc2.weight"].detach().cpu().numpy(),
        "b2": state["fc2.bias"].detach().cpu().numpy(),
        "w_flip": state["flip_head.weight"].detach().cpu().numpy(),
        "b_flip": state["flip_head.bias"].detach().cpu().numpy(),
    }

    if "budget_head.weight" in state:
        out["w_budget"] = state["budget_head.weight"].detach().cpu().numpy()
        out["b_budget"] = state["budget_head.bias"].detach().cpu().numpy()
        out["variant"] = "budgeted"
    else:
        out["w_budget"] = None
        out["b_budget"] = None
        out["variant"] = "mask_only"

    return out
