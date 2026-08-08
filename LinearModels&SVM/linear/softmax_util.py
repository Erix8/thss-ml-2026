import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def train_multiclass_softmax_with_model_selection(
    X_train, y_train, X_val, y_val,
    num_classes=4,
    lr=1e-2,
    epochs=3000,
    batch_size=64,
    weight_decay=0.0,
    verbose_every=100,
    select_by="val_loss",   # "val_loss" or "val_acc"
):
    device = torch.device("cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    n_train, d = X_train.shape
    model = nn.Linear(d, num_classes, bias=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    if select_by == "val_loss":
        best_metric = float("inf")
    elif select_by == "val_acc":
        best_metric = -float("inf")
    else:
        raise ValueError("select_by must be 'val_loss' or 'val_acc'")

    best_epoch = -1
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()

        perm = torch.randperm(n_train, device=device)
        X_train_epoch = X_train_t[perm]
        y_train_epoch = y_train_t[perm]

        batch_losses = []
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            xb = X_train_epoch[start:end]
            yb = y_train_epoch[start:end]

            optimizer.zero_grad()
            logits = model(xb)          # (bs, C)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        history["train_loss"].append(train_loss)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_pred = torch.argmax(val_logits, dim=1)
            val_acc = (val_pred == y_val_t).float().mean().item()

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = False
        if select_by == "val_loss" and val_loss < best_metric:
            best_metric = val_loss
            improved = True
        elif select_by == "val_acc" and val_acc > best_metric:
            best_metric = val_acc
            improved = True

        if improved:
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        if epoch % verbose_every == 0 or epoch == 1 or epoch == epochs:
            print(f"[Softmax] epoch={epoch:4d} train_loss={train_loss:.6f} "
                  f"val_loss={val_loss:.6f} val_acc={val_acc:.4f} "
                  f"best_epoch={best_epoch}")

    model.load_state_dict(best_state_dict)

    print("\n===== Best Model Selected =====")
    if select_by == "val_loss":
        print(f"criterion=val_loss, best_epoch={best_epoch}, best_val_loss={best_metric:.6f}")
    else:
        print(f"criterion=val_acc, best_epoch={best_epoch}, best_val_acc={best_metric:.4f}")

    W = model.weight.detach().cpu().numpy()  # (C, d)
    for c in range(num_classes):
        print(f"class {c} weights: {W[c]}")

    return model, history, {"best_epoch": best_epoch, "best_metric": best_metric, "select_by": select_by}


def evaluate_multiclass_softmax(model, X_test, y_test):
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()

    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        loss = criterion(logits, y_test_t).item()
        pred = torch.argmax(logits, dim=1)
        acc = (pred == y_test_t).float().mean().item()

    print(f"[Test(best model)] loss={loss:.6f}, acc={acc:.4f}")
    return loss, acc