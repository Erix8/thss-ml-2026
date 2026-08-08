import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from softmax_util import (
    train_multiclass_softmax_with_model_selection,
    evaluate_multiclass_softmax,
)


def split_data(X, y1, y2, y3, split_size=[0.8, 0.2], shuffle=False, random_seed=None):
    """Split the dataset into train/val/test subsets.

    Args:
        X - feature matrix, 2D numpy array of shape (num_instances, num_features)
        yi - label vector, 1D numpy array of shape (num_instances,)
        split_size - split ratios, e.g. [0.8, 0.2] means 80% train and 20% test
        shuffle - whether to shuffle the dataset
        random_seed - random seed for reproducibility

    Returns:
        X_list - list of feature sub-arrays
        yi_list - list of label sub-arrays
    """
    assert sum(split_size) == 1
    num_instances = X.shape[0]
    if shuffle:
        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(num_instances)
        X = X[indices]
        y1 = y1[indices]
        y2 = y2[indices]
        y3 = y3[indices]

    # TODO 2.1.1 (about 7 lines)
    # compute split point indices and use np.split to partition the dataset
    split_points = (np.cumsum(split_size)[:-1] * num_instances).astype(int)
    X_list = np.split(X, split_points)
    y1_list = np.split(y1, split_points)
    y2_list = np.split(y2, split_points)
    y3_list = np.split(y3, split_points)

    return X_list, y1_list, y2_list, y3_list


def feature_normalization(train, val, test):
    """Map all features in the training set to [0,1]; apply the same affine transform to val/test.

    Args:
        train - training set, 2D numpy array of shape (num_instances, num_features)
        val - validation set, 2D numpy array of shape (num_instances, num_features)
        test - test set, 2D numpy array of shape (num_instances, num_features)
    Returns:
        train_normalized - normalized training set
        val_normalized - normalized validation set
        test_normalized - normalized test set

    """
    # TODO 2.1.2 (about 8 lines)
    # compute min and max of each feature on the training set
    train_min = np.min(train, axis=0)
    train_max = np.max(train, axis=0)
    # compute feature range, avoid division by zero
    train_range = train_max - train_min
    train_range[train_range == 0] = 1.0
    # normalize train, val, test using train min and range
    train_normalized = (train - train_min) / train_range
    val_normalized = (val - train_min) / train_range
    test_normalized = (test - train_min) / train_range

    return train_normalized, val_normalized, test_normalized


def build_basis_features(X_raw):
    """
    Construct 9 basis functions using only x1, x2, x3:
    [x1, x2, x3, cos(x1), cos(x2), cos(x3), x1^2, x2^2, x3^2]
    X_raw: shape (n, num_features), assumes first 3 columns are x1,x2,x3

    Returns
    -------
    Phi : numpy.ndarray, shape (n_samples, 9), dtype=float
        Design matrix after basis expansion. Column order fixed as:
        1) x1
        2) x2
        3) x3
        4) cos(x1)
        5) cos(x2)
        6) cos(x3)
        7) x1^2
        8) x2^2
        9) x3^2

    feature_names : list[str], length=9
        Names corresponding to each column of Phi, e.g.:
        ["x1", "x2", "x3", "cos(x1)", "cos(x2)", "cos(x3)", "x1^2", "x2^2", "x3^2"]
    """
    x1 = X_raw[:, 0]
    x2 = X_raw[:, 1]
    x3 = X_raw[:, 2]

    # TODO 2.2 basis functions (3~10 lines)
    # use np.column_stack to build basis matrix Phi and define feature_names list
    Phi = np.column_stack(
        [x1, x2, x3, np.cos(x1), np.cos(x2), np.cos(x3), x1**2, x2**2, x3**2]
    )
    feature_names = [
        "x1",
        "x2",
        "x3",
        "cos(x1)",
        "cos(x2)",
        "cos(x3)",
        "x1^2",
        "x2^2",
        "x3^2",
    ]

    return Phi, feature_names


def train_linear_with_regularization(
    X_train,
    y_train,
    X_val,
    y_val,
    reg_type="l2",  # "l1" or "l2"
    lambda_reg=1e-2,
    lr=1e-2,
    epochs=5000,
    verbose_every=500,
):
    """Linear model: y = w^T x + b. Loss: MSE + lambda_reg * (L1 or L2 regularization)."""

    device = torch.device("cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32, device=device)

    model = nn.Linear(X_train.shape[1], 1, bias=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred = model(X_train_t)
        data_loss = mse(pred, y_train_t)

        w = model.weight  # shape (1, d)
        if reg_type == "l1":
            reg_loss = torch.abs(w).sum()
        elif reg_type == "l2":
            reg_loss = (w**2).sum()
        else:
            raise ValueError("reg_type must be 'l1' or 'l2'")

        loss = data_loss + lambda_reg * reg_loss
        loss.backward()
        optimizer.step()

        if epoch % verbose_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_mse = mse(val_pred, y_val_t).item()
            print(
                f"[{reg_type}] epoch={epoch:4d} train_obj={loss.item():.6f} val_mse={val_mse:.6f}"
            )

    # output converged weights
    w_final = model.weight.detach().cpu().numpy().reshape(-1)
    b_final = float(model.bias.detach().cpu().numpy().reshape(-1)[0])

    return model, w_final, b_final


def compute_regularized_square_loss(X, y, theta, lambda_reg):
    """
    Compute ridge regression loss using X*theta to predict y.

    Args:
        X - feature matrix, array of shape (num_instances, num_features)
        y - label vector, array of shape (num_instances,)
        theta - parameter vector, array of shape (num_features,)
        lambda_reg - regularization coefficient

    Returns:
        loss - scalar loss value
    """
    # TODO 2.3.2 (2~7 lines)
    m = X.shape[0]
    # compute residual vector
    residual = X @ theta - y
    # compute data loss and regularization loss, return total loss
    data_loss = (residual.T @ residual) / m
    reg_loss = lambda_reg * (theta.T @ theta)
    loss = data_loss + reg_loss
    return float(loss)


def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute gradient of the ridge regression loss function.

    Args:
        X - feature matrix, array of shape (num_instances, num_features)
        y - label vector, array of shape (num_instances,)
        theta - parameter vector, array of shape (num_features,)
        lambda_reg - regularization coefficient

    Returns:
        grad - gradient vector, array of shape (num_features,)
    """
    # TODO 2.3.4 (2~7 lines)
    m = X.shape[0]
    residual = X @ theta - y
    grad_data = (2.0 / m) * (X.T @ residual)
    grad_reg = 2.0 * lambda_reg * theta
    grad = grad_data + grad_reg
    return grad


def grad_checker(X, y, theta, lambda_reg, epsilon=0.01, tolerance=1e-4):
    """Gradient check: verifies analytical gradient against numerical approximation.
    If the Euclidean distance between actual and approximate gradients exceeds
    the tolerance, the gradient computation is incorrect.

    Args:
        X - feature matrix, array of shape (num_instances, num_features)
        y - label vector, array of shape (num_instances,)
        theta - parameter vector, array of shape (num_features,)
        lambda_reg - regularization coefficient
        epsilon - step size for numerical gradient
        tolerance - tolerance for gradient difference

    Returns:
        True if gradient is correct, False otherwise

    """
    grad_computed = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
    num_features = theta.shape[0]
    grad_approx = np.zeros(num_features)

    for h in np.identity(num_features):
        J0 = compute_regularized_square_loss(X, y, theta - epsilon * h, lambda_reg)
        J1 = compute_regularized_square_loss(X, y, theta + epsilon * h, lambda_reg)
        grad_approx += (J1 - J0) / (2 * epsilon) * h
    dist = np.linalg.norm(grad_approx - grad_computed)
    return dist <= tolerance


def grad_descent(X, y, lambda_reg, alpha=0.1, num_iter=1000, check_gradient=False):
    """Full-batch gradient descent algorithm.

    Args:
        X - feature matrix, array of shape (num_instances, num_features)
        y - label vector, array of shape (num_instances,)
        lambda_reg - regularization coefficient
        alpha - gradient descent step size
        num_iter - number of iterations to run
        check_gradient - whether to check gradient correctness at each update

    Returns:
        theta_hist - history of parameter vectors, 2D array of shape (num_iter+1, num_features)
        loss_hist - history of full-batch loss values, 1D array of shape (num_iter,)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist
    for i in range(num_iter):
        # TODO 2.4.2 (3~5 lines)
        if check_gradient:
            assert grad_checker(X, y, theta, lambda_reg), "Gradient check failed"
        grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        theta = theta - alpha * grad
        theta_hist[i + 1] = theta
        loss_hist[i] = compute_regularized_square_loss(X, y, theta, lambda_reg)

    return theta_hist, loss_hist


def stochastic_grad_descent(
    X_train, y_train, X_val, y_val, lambda_reg, alpha=0.1, num_iter=1000, batch_size=1
):
    """Stochastic gradient descent with validation monitoring.

    Args:
        X_train - training feature matrix, array of shape (num_instances, num_features)
        y_train - training label vector, array of shape (num_instances,)
        X_val - validation feature matrix, array of shape (num_instances, num_features)
        y_val - validation label vector, array of shape (num_instances,)
        alpha - gradient descent step size
        lambda_reg - regularization coefficient
        num_iter - number of iterations to run
        batch_size - mini-batch size

    Returns:
        theta_hist - history of parameter vectors, 2D array of shape (num_iter+1, num_features)
        loss_hist - history of mini-batch regularized loss, array of shape (num_iter,)
        validation_hist - history of full-batch MSE on validation set (no reg), array of shape (num_iter,)
    """
    num_instances, num_features = X_train.shape[0], X_train.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist
    validation_hist = np.zeros(num_iter)  # Initialize validation_hist

    # TODO 2.6.2
    for i in range(num_iter):
        replace = batch_size > num_instances
        batch_indices = np.random.choice(
            num_instances, size=batch_size, replace=replace
        )
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        residual = X_batch @ theta - y_batch
        grad_data = (2.0 / X_batch.shape[0]) * (X_batch.T @ residual)
        grad_reg = 2.0 * lambda_reg * theta
        grad = grad_data + grad_reg

        theta = theta - alpha * grad
        theta_hist[i + 1] = theta

        loss_hist[i] = compute_regularized_square_loss(
            X_batch, y_batch, theta, lambda_reg
        )
        validation_hist[i] = np.mean((X_val @ theta - y_val) ** 2)

    return theta_hist, loss_hist, validation_hist


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run selected experiments for the HW1 linear assignment."
    )
    parser.add_argument(
        "--basis-reg",
        action="store_true",
        help="Run basis features and regularization experiments.",
    )
    parser.add_argument(
        "--grad-check",
        action="store_true",
        help="Run the regularized square-loss gradient check.",
    )
    parser.add_argument(
        "--gd-search",
        action="store_true",
        help="Run full-batch gradient descent and hyper-parameter search.",
    )
    parser.add_argument(
        "--sgd",
        action="store_true",
        help="Run stochastic gradient descent experiments.",
    )
    parser.add_argument(
        "--classification",
        action="store_true",
        help="Run the softmax multi-class classification experiment.",
    )
    return parser


def load_dataset_splits():
    print("loading the dataset")

    df = pd.read_csv("train.csv", delimiter=",")
    X = df.values[:, :-3]
    y1 = df.values[:, -3]
    y2 = df.values[:, -2]
    y3 = df.values[:, -1]

    print("Split into Train and Val")
    (
        (X_train_raw, X_val_raw),
        (y1_train, y1_val),
        (y2_train, y2_val),
        (y3_train, y3_val),
    ) = split_data(X, y1, y2, y3, split_size=[0.8, 0.2], shuffle=True, random_seed=0)

    df_test = pd.read_csv("test.csv", delimiter=",")
    X_test_raw = df_test.values[:, :-3]
    y1_test = df_test.values[:, -3]
    y2_test = df_test.values[:, -2]
    y3_test = df_test.values[:, -1]

    return (
        X_train_raw,
        X_val_raw,
        X_test_raw,
        y1_train,
        y1_val,
        y1_test,
        y2_train,
        y2_val,
        y2_test,
        y3_train,
        y3_val,
        y3_test,
    )


def prepare_linear_features(X_train_raw, X_val_raw, X_test_raw):
    print("Scaling all to [0, 1]")
    X_train, X_val, X_test = feature_normalization(X_train_raw, X_val_raw, X_test_raw)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
    X_val = np.hstack((X_val, np.ones((X_val.shape[0], 1))))
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
    return X_train, X_val, X_test


def run_basis_and_regularization_experiment(
    X_train_raw,
    X_val_raw,
    X_test_raw,
    y1_train,
    y1_val,
):
    Phi_train, feature_names = build_basis_features(X_train_raw[:, :3])
    Phi_val, _ = build_basis_features(X_val_raw[:, :3])
    Phi_test, _ = build_basis_features(X_test_raw[:, :3])
    Phi_train, Phi_val, Phi_test = feature_normalization(Phi_train, Phi_val, Phi_test)

    target_train = y1_train
    target_val = y1_val
    lambda_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 5]

    weight_columns = [
        "lambda",
        "x1",
        "x2",
        "x3",
        "cosx1",
        "cosx2",
        "cosx3",
        "x1_sq",
        "x2_sq",
        "x3_sq",
    ]

    print("\n===== L2 Regularization =====")
    l2_weight_rows = []
    for lam in lambda_list:
        print(f"\n--- lambda={lam} ---")
        _, w_l2, b_l2 = train_linear_with_regularization(
            Phi_train,
            target_train,
            Phi_val,
            target_val,
            reg_type="l2",
            lambda_reg=lam,
            lr=1e-3,
            epochs=10000,
            verbose_every=1000,
        )
        print("bias:", b_l2)
        print("weights:")
        for n, w in zip(feature_names, w_l2):
            print(f"  {n:8s}: {w:+.6f}")

        l2_weight_rows.append(
            {
                "lambda": lam,
                "x1": w_l2[0],
                "x2": w_l2[1],
                "x3": w_l2[2],
                "cosx1": w_l2[3],
                "cosx2": w_l2[4],
                "cosx3": w_l2[5],
                "x1_sq": w_l2[6],
                "x2_sq": w_l2[7],
                "x3_sq": w_l2[8],
            }
        )

    pd.DataFrame(l2_weight_rows, columns=weight_columns).to_csv(
        "l2_weights.csv", index=False
    )
    print("saved L2 weights to l2_weights.csv")

    print("\n===== L1 Regularization =====")
    l1_weight_rows = []
    for lam in lambda_list:
        print(f"\n--- lambda={lam} ---")
        _, w_l1, b_l1 = train_linear_with_regularization(
            Phi_train,
            target_train,
            Phi_val,
            target_val,
            reg_type="l1",
            lambda_reg=lam,
            lr=1e-3,
            epochs=10000,
            verbose_every=1000,
        )
        print("bias:", b_l1)
        print("weights:")
        for n, w in zip(feature_names, w_l1):
            print(f"  {n:8s}: {w:+.6f}")

        l1_weight_rows.append(
            {
                "lambda": lam,
                "x1": w_l1[0],
                "x2": w_l1[1],
                "x3": w_l1[2],
                "cosx1": w_l1[3],
                "cosx2": w_l1[4],
                "cosx3": w_l1[5],
                "x1_sq": w_l1[6],
                "x2_sq": w_l1[7],
                "x3_sq": w_l1[8],
            }
        )

    pd.DataFrame(l1_weight_rows, columns=weight_columns).to_csv(
        "l1_weights.csv", index=False
    )
    print("saved L1 weights to l1_weights.csv")


def run_gradient_check_experiment():
    print("===== Gradient Check for Regularized Square Loss =====")
    rng = np.random.RandomState(0)
    X_gc = rng.randn(50, 6)
    y_gc = rng.randn(50)
    theta_gc = rng.randn(6)
    lambda_gc = 1e-2
    is_grad_correct = grad_checker(X_gc, y_gc, theta_gc, lambda_gc)
    print(f"Gradient check passed: {is_grad_correct}")


def run_grad_descent_search_experiment(
    X_train,
    X_val,
    X_test,
    y2_train,
    y2_val,
    y2_test,
):
    print("\n===== Hyper-parameter Search for grad_descent (target=y2) =====")
    target_train = y2_train
    target_val = y2_val

    alpha_list = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    lambda_list = [0, 1e-5, 1e-4, 1e-3]
    num_iter = 5000
    sample_every = 100
    sampled_epochs = np.unique(
        np.concatenate(
            ([1], np.arange(sample_every, num_iter + 1, sample_every), [num_iter])
        )
    )

    results = []
    curve_records = []
    curve_series = []
    run_id = 0
    for alpha in alpha_list:
        for lambda_reg in lambda_list:
            run_id += 1
            theta_hist, loss_hist = grad_descent(
                X_train,
                target_train,
                lambda_reg=lambda_reg,
                alpha=alpha,
                num_iter=num_iter,
                check_gradient=True,
            )

            theta_final = theta_hist[-1]
            train_pred = X_train @ theta_final
            val_pred = X_val @ theta_final
            train_mse = np.mean((train_pred - target_train) ** 2)
            val_mse = np.mean((val_pred - target_val) ** 2)

            result = {
                "run": run_id,
                "alpha": alpha,
                "lambda_reg": lambda_reg,
                "final_obj": float(loss_hist[-1]),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
                "theta": theta_final.copy(),
            }
            results.append(result)

            sampled_val_mse = []
            for epoch in sampled_epochs:
                theta_epoch = theta_hist[epoch]
                train_mse_epoch = np.mean((X_train @ theta_epoch - target_train) ** 2)
                val_mse_epoch = np.mean((X_val @ theta_epoch - target_val) ** 2)
                sampled_val_mse.append(float(val_mse_epoch))
                curve_records.append(
                    {
                        "alpha": float(alpha),
                        "lambda_reg": float(lambda_reg),
                        "epoch": int(epoch),
                        "train_mse": float(train_mse_epoch),
                        "val_mse": float(val_mse_epoch),
                    }
                )

            curve_series.append(
                {
                    "label": f"a={alpha:.4g}, l={lambda_reg:.4g}",
                    "epochs": sampled_epochs.copy(),
                    "val_mse": np.array(sampled_val_mse),
                }
            )

            print(
                "run={run:02d} alpha={alpha:.4g} lambda={lambda_reg:.4g} "
                "final_obj={final_obj:.6f} train_mse={train_mse:.6f} val_mse={val_mse:.6f}".format(
                    **result
                )
            )

    best_result = min(results, key=lambda r: r["val_mse"])
    print("\nBest hyper-parameters by validation MSE:")
    print(
        "alpha={:.4g}, lambda={:.4g}, val_mse={:.6f}, train_mse={:.6f}, final_obj={:.6f}".format(
            best_result["alpha"],
            best_result["lambda_reg"],
            best_result["val_mse"],
            best_result["train_mse"],
            best_result["final_obj"],
        )
    )

    best_theta = best_result["theta"]
    test_pred = X_test @ best_theta
    test_mse = np.mean((test_pred - y2_test) ** 2)
    print("Best model test_mse={:.6f}".format(test_mse))

    curve_df = pd.DataFrame(curve_records)
    curve_df.to_csv("gd_hparam_curve_samples.csv", index=False)

    plt.figure(figsize=(10, 6))
    for series in curve_series:
        plt.plot(
            series["epochs"], series["val_mse"], linewidth=1.2, label=series["label"]
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.title("Grad Descent Hyper-parameter Search: Validation MSE Curves")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig("gd_hparam_all_curves.png", dpi=300)
    plt.close()

    print("Saved sampled MSE records to gd_hparam_curve_samples.csv")
    print("Saved all hyper-parameter curves to gd_hparam_all_curves.png")


def run_sgd_experiment(
    X_train,
    X_val,
    X_test,
    y2_train,
    y2_val,
    y2_test,
):
    print("\n===== SGD Experiments (target=y2, alpha=0.1, lambda=0) =====")
    target_train = y2_train
    target_val = y2_val

    alpha = 0.03
    lambda_reg = 0.0
    num_iter = 5000
    batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128]
    if num_iter <= 1000:
        sample_epochs = np.unique(
            np.linspace(1, num_iter, num=min(num_iter, 100), dtype=int)
        )
    else:
        sample_epochs_low = np.unique(np.linspace(1, 999, num=50, dtype=int))
        sample_epochs_high = np.unique(np.linspace(1001, num_iter, num=50, dtype=int))
        sample_epochs = np.unique(
            np.concatenate((sample_epochs_low, sample_epochs_high, [num_iter]))
        )

    np.random.seed(0)
    sgd_results = []
    sgd_curve_samples = []
    for batch_size in batch_size_list:
        theta_hist, loss_hist, validation_hist = stochastic_grad_descent(
            X_train,
            target_train,
            X_val,
            target_val,
            lambda_reg=lambda_reg,
            alpha=alpha,
            num_iter=num_iter,
            batch_size=batch_size,
        )

        theta_final = theta_hist[-1]
        train_mse = np.mean((X_train @ theta_final - target_train) ** 2)
        val_mse = np.mean((X_val @ theta_final - target_val) ** 2)

        result = {
            "batch_size": batch_size,
            "final_obj": float(loss_hist[-1]),
            "final_val_mse": float(validation_hist[-1]),
            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
            "theta": theta_final.copy(),
        }
        sgd_results.append(result)
        print(
            "batch_size={batch_size:3d} final_obj={final_obj:.6f} "
            "final_val_mse={final_val_mse:.6f} train_mse={train_mse:.6f} "
            "val_mse={val_mse:.6f}".format(**result)
        )
        for epoch in sample_epochs:
            theta_epoch = theta_hist[epoch]
            train_mse_epoch = np.mean((X_train @ theta_epoch - target_train) ** 2)
            val_mse_epoch = validation_hist[epoch - 1]
            obj_epoch = loss_hist[epoch - 1]
            sgd_curve_samples.append(
                {
                    "batch_size": batch_size,
                    "epoch": epoch,
                    "train_mse": float(train_mse_epoch),
                    "val_mse": float(val_mse_epoch),
                    "batch_obj": float(obj_epoch),
                }
            )

    best_result = min(sgd_results, key=lambda r: r["val_mse"])
    print("\nBest SGD setting by validation MSE:")
    print(
        "batch_size={batch_size}, alpha={alpha}, lambda={lambda_reg}, "
        "val_mse={val_mse:.6f}".format(
            batch_size=best_result["batch_size"],
            alpha=alpha,
            lambda_reg=lambda_reg,
            val_mse=best_result["val_mse"],
        )
    )

    best_theta = best_result["theta"]
    best_test_mse = np.mean((X_test @ best_theta - y2_test) ** 2)
    print("Best batch-size model test_mse={:.6f}".format(best_test_mse))

    sgd_summary_df = pd.DataFrame(sgd_results).drop(columns=["theta"])
    sgd_summary_df.to_csv("sgd_batchsize_results.csv", index=False)
    pd.DataFrame(sgd_curve_samples).to_csv("sgd_curve_samples.csv", index=False)


def run_classification_experiment(X_train, X_val, X_test, y3_train, y3_val, y3_test):
    print("\n===== y3 Multi-class Classification (Softmax + Model Selection) =====")
    y3_train_cls = y3_train.astype(np.int64)
    y3_val_cls = y3_val.astype(np.int64)
    y3_test_cls = y3_test.astype(np.int64)

    best_model, clf_hist, best_info = train_multiclass_softmax_with_model_selection(
        X_train=X_train,
        y_train=y3_train_cls,
        X_val=X_val,
        y_val=y3_val_cls,
        num_classes=4,
        lr=1e-3,
        epochs=1000,
        batch_size=20,
        weight_decay=1e-4,
        verbose_every=100,
        select_by="val_acc",  # "val_acc" or "val_loss"
    )

    evaluate_multiclass_softmax(best_model, X_test, y3_test_cls)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if not any(
        [
            args.basis_reg,
            args.grad_check,
            args.gd_search,
            args.sgd,
            args.classification,
        ]
    ):
        parser.print_help()
        return

    (
        X_train_raw,
        X_val_raw,
        X_test_raw,
        y1_train,
        y1_val,
        y1_test,
        y2_train,
        y2_val,
        y2_test,
        y3_train,
        y3_val,
        y3_test,
    ) = load_dataset_splits()

    X_train, X_val, X_test = prepare_linear_features(X_train_raw, X_val_raw, X_test_raw)

    if args.basis_reg:
        run_basis_and_regularization_experiment(
            X_train_raw,
            X_val_raw,
            X_test_raw,
            y1_train,
            y1_val,
        )

    if args.grad_check:
        run_gradient_check_experiment()

    if args.gd_search:
        run_grad_descent_search_experiment(
            X_train,
            X_val,
            X_test,
            y2_train,
            y2_val,
            y2_test,
        )

    if args.sgd:
        run_sgd_experiment(
            X_train,
            X_val,
            X_test,
            y2_train,
            y2_val,
            y2_test,
        )

    if args.classification:
        run_classification_experiment(
            X_train,
            X_val,
            X_test,
            y3_train,
            y3_val,
            y3_test,
        )


if __name__ == "__main__":
    main()
