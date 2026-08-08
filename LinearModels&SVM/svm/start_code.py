import pandas as pd
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import trange
import matplotlib.pyplot as plt


def accuracy_score(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] == 0:
        return np.nan
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, labels=None):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    labels = np.asarray(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    cm = np.zeros((labels.shape[0], labels.shape[0]), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def f1_score(y_true, y_pred, pos_label=1):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_text_dataset(filename, positive="joy", negative="sadness"):
    """Read text dataset from file."""
    data = pd.read_csv(filename)
    is_positive = data.Emotion == positive
    is_negative = data.Emotion == negative
    data = data[is_positive | is_negative]
    X = data.Text  # input text
    y = np.array(data.Emotion == positive) * 2 - 1  # 1: positive, -1: negative
    return X, y


def vectorize(train, val):
    """Convert training and validation text data to vector representations.

    Args:
        train - training set, array of text strings, shape (num_instances,)
        val - test/validation set, array of text strings, shape (num_instances,)
    Returns:
        train_normalized - vectorized training set (num_instances, num_features)
        val_normalized - vectorized validation set (num_instances, num_features)
    """
    tfidf = TfidfVectorizer(stop_words="english", use_idf=True, smooth_idf=True)
    train_normalized = tfidf.fit_transform(train).toarray()
    val_normalized = tfidf.transform(val).toarray()
    return train_normalized, val_normalized


def gaussian_kernel_matrix(X1, X2, gamma=0.1):
    """Compute Gaussian (RBF) kernel matrix: K(i, j) = exp(-gamma * ||x_i - x_j||^2)."""
    sq1 = np.sum(X1**2, axis=1).reshape(-1, 1)
    sq2 = np.sum(X2**2, axis=1).reshape(1, -1)
    dist_sq = sq1 + sq2 - 2 * X1 @ X2.T
    return np.exp(-gamma * dist_sq)


def linear_svm_subgrad_descent(
    X,
    y,
    alpha=0.05,
    lambda_reg=0.0001,
    num_iter=60000,
    batch_size=16,
    random_seed=42,
    X_val=None,
    y_val=None,
    eval_every=100,
):
    """Stochastic subgradient descent for linear SVM.

    Args:
        X - feature matrix, array of shape (num_instances, num_features)
        y - label vector, array of shape (num_instances,)
        alpha - float, gradient descent step size
        lambda_reg - regularization coefficient
        num_iter - number of iterations to run
        batch_size - mini-batch size
        random_seed - seed for initializing random sampling
        X_val - validation feature matrix, array of shape (num_val_instances, num_features), optional
        y_val - validation label vector, array of shape (num_val_instances,), optional
        eval_every - evaluate validation loss every eval_every iterations

    Returns:
        theta_hist - history of parameter vectors, 2D array of shape (num_iter+1, num_features)
        loss_hist - history of mini-batch loss values, array of shape (num_iter,)
        val_loss_hist - history of full-batch validation loss, array of shape (num_iter,)
    """

    # add bias term
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    X_val = np.hstack((X_val, np.ones((X_val.shape[0], 1))))

    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist
    val_loss_hist = np.full(num_iter, np.nan)  # Initialize val_loss_hist

    # TODO 3.5.1
    rng = np.random.default_rng(random_seed)
    for i in trange(num_iter):
        replace = batch_size > num_instances
        batch_idx = rng.choice(num_instances, size=batch_size, replace=replace)
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]

        # compute hinge loss margins and hinge loss
        margins = y_batch * (X_batch @ theta)
        hinge = np.maximum(0.0, 1.0 - margins)

        # compute subgradient of hinge loss
        violators = margins < 1.0
        if np.any(violators):
            grad_hinge = -np.mean(y_batch[violators, None] * X_batch[violators], axis=0)
        else:
            grad_hinge = np.zeros_like(theta)

        # compute total gradient and update parameters
        grad = lambda_reg * theta + grad_hinge
        theta = theta - alpha * grad
        theta_hist[i + 1] = theta
        loss_hist[i] = 0.5 * lambda_reg * np.dot(theta, theta) + np.mean(hinge)

        # evaluate validation set loss
        if (
            X_val is not None
            and y_val is not None
            and ((i % eval_every == 0) or (i == num_iter - 1))
        ):
            val_margins = y_val * (X_val @ theta)
            val_hinge = np.maximum(0.0, 1.0 - val_margins)
            val_loss_hist[i] = 0.5 * lambda_reg * np.dot(theta, theta) + np.mean(
                val_hinge
            )

    return theta_hist, loss_hist, val_loss_hist


def kernel_svm_subgrad_descent(
    X,
    y,
    alpha=0.1,
    lambda_reg=0.0001,
    num_iter=6000,
    batch_size=16,
    gamma=0.1,
    random_seed=42,
    X_val=None,
    y_val=None,
    eval_every=100,
):
    """Stochastic subgradient descent for kernel SVM.

    Args:
        X - feature matrix, array of shape (num_instances, num_features)
        y - label vector, array of shape (num_instances,)
        alpha - float, initial gradient descent step size
        lambda_reg - regularization coefficient
        num_iter - number of passes over the entire training set
        batch_size - mini-batch size
        gamma - RBF kernel parameter
        random_seed - seed for initializing the random sampling
        X_val - validation feature matrix, array of shape (num_val_instances, num_features), optional
        y_val - validation label vector, array of shape (num_val_instances,), optional
        eval_every - evaluate validation loss every eval_every iterations

    Returns:
        theta_hist - history of parameter vectors, 2D array of shape (num_iter+1, num_features+1)
        loss_hist - history of mini-batch loss values, array of shape (num_iter,)
        val_loss_hist - history of full-batch validation loss, array of shape (num_iter,)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_instances + 1)  # Initialize theta = [alpha_vec, b]
    theta_hist = np.zeros((num_iter + 1, num_instances + 1))  # Initialize theta_hist
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist
    val_loss_hist = np.full(num_iter, np.nan)  # Initialize val_loss_hist

    # TODO 3.5.2
    alpha_vec = np.zeros(num_instances)
    b = 0.0
    theta_hist[0] = np.concatenate([alpha_vec, np.array([b])])
    rng = np.random.default_rng(random_seed)

    # precompute kernel matrix to avoid redundant calculation
    K_train = gaussian_kernel_matrix(X, X, gamma=gamma)
    K_val_train = None
    if X_val is not None and y_val is not None:
        K_val_train = gaussian_kernel_matrix(X_val, X, gamma=gamma)

    for i in trange(num_iter):
        replace = batch_size > num_instances
        batch_idx = rng.choice(num_instances, size=batch_size, replace=replace)
        y_batch = y[batch_idx]
        K_batch_train = K_train[batch_idx]

        # compute hinge loss margins and hinge loss
        decision_batch = K_batch_train @ (alpha_vec * y) + b
        margins = y_batch * decision_batch
        hinge = np.maximum(0.0, 1.0 - margins)

        # only violating samples (margin < 1) contribute to the hinge subgradient
        violators = margins < 1.0
        if np.any(violators):
            y_viol = y_batch[violators]
            K_viol = K_batch_train[violators]  # (num_violators, num_instances)
            grad_hinge_alpha = -(y * (K_viol.T @ y_viol)) / batch_size
            grad_hinge_b = -np.sum(y_viol) / batch_size
        else:
            grad_hinge_alpha = np.zeros_like(alpha_vec)
            grad_hinge_b = 0.0

        # regularization only applies to alpha weights; gradient for bias b is 0
        grad_reg_alpha = lambda_reg * y * (K_train @ (alpha_vec * y))
        grad_alpha = grad_reg_alpha + grad_hinge_alpha
        grad_b = grad_hinge_b

        # update parameters
        alpha_vec = alpha_vec - alpha * grad_alpha
        b = b - alpha * grad_b
        theta = np.concatenate([alpha_vec, np.array([b])])
        theta_hist[i + 1] = theta

        # current batch loss
        decision_batch = K_batch_train @ (alpha_vec * y) + b
        margins = y_batch * decision_batch
        hinge = np.maximum(0.0, 1.0 - margins)
        reg_loss = 0.5 * lambda_reg * (alpha_vec * y) @ K_train @ (alpha_vec * y)
        loss_hist[i] = reg_loss + np.mean(hinge)

        # validation set full-batch loss
        if K_val_train is not None and ((i % eval_every == 0) or (i == num_iter - 1)):
            val_decision = K_val_train @ (alpha_vec * y) + b
            val_margins = y_val * val_decision
            val_hinge = np.maximum(0.0, 1.0 - val_margins)
            val_loss_hist[i] = reg_loss + np.mean(val_hinge)

    return theta_hist, loss_hist, val_loss_hist


def main():
    parser = argparse.ArgumentParser(
        description="Train linear SVM and/or kernel SVM on the emotion dataset."
    )
    parser.add_argument(
        "--linear-svm",
        action="store_true",
        help="train the linear SVM model",
    )
    parser.add_argument(
        "--kernel-svm",
        action="store_true",
        help="train the kernel SVM model",
    )
    args = parser.parse_args()

    run_linear_svm = args.linear_svm
    run_kernel_svm = args.kernel_svm

    if not run_linear_svm and not run_kernel_svm:
        parser.print_help()
        return

    # load all data, convert train/val text to vector representations
    X_train, y_train = load_text_dataset("data_train.csv", "joy", "sadness")
    X_val, y_val = load_text_dataset("data_val.csv")
    print(
        "Training set size: {}, Validation set size: {}".format(
            len(X_train), len(X_val)
        )
    )
    X_train_vect, X_val_vect = vectorize(X_train, X_val)

    # check if at least one model is to be trained
    if not run_linear_svm and not run_kernel_svm:
        print("Both run_linear_svm and run_kernel_svm are False. Nothing to train.")
        return

    # linear SVM stochastic subgradient descent training
    if run_linear_svm:
        # hyperparameters can be tuned for better validation performance
        linear_alpha = 0.05
        linear_lambda = 0.0001
        linear_batch_size = 32
        linear_num_iter = 30000
        linear_eval_every = 200

        print(
            "[Linear SVM] Training with alpha={}, lambda_reg={}, batch_size={}".format(
                linear_alpha, linear_lambda, linear_batch_size
            )
        )
        theta_hist, loss_hist, val_loss_hist = linear_svm_subgrad_descent(
            X_train_vect,
            y_train,
            alpha=linear_alpha,
            lambda_reg=linear_lambda,
            num_iter=linear_num_iter,
            batch_size=linear_batch_size,
            X_val=X_val_vect,
            y_val=y_val,
            eval_every=linear_eval_every,
        )
        theta = theta_hist[-1]

        # compute accuracy, F1-score, and confusion matrix of linear SVM on validation set
        y_val_scores = X_val_vect @ theta[:-1] + theta[-1]
        y_val_pred = np.where(y_val_scores >= 0, 1, -1)
        accuracy = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred, pos_label=1)
        cm = confusion_matrix(y_val, y_val_pred, labels=[-1, 1])

        print("[Linear SVM]")
        print("Validation Accuracy: {:.4f}".format(accuracy))
        print("Validation F1-Score: {:.4f}".format(f1))
        print("Confusion Matrix [[TN, FP], [FN, TP]]:")
        print(cm)

        # plot linear SVM validation loss curve
        plt.figure(figsize=(8, 4))
        linear_eval_idx = np.where(~np.isnan(val_loss_hist))[0]
        plt.plot(
            linear_eval_idx,
            val_loss_hist[linear_eval_idx],
            marker="o",
            markersize=2,
            label="Validation Loss",
        )
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Linear SVM Validation Loss")

        linear_val_loss_df = pd.DataFrame(
            {
                "epoch": linear_eval_idx + 1,
                "val_loss": val_loss_hist[linear_eval_idx],
                "alpha": linear_alpha,
                "lambda_reg": linear_lambda,
                "batch_size": linear_batch_size,
            }
        )
        linear_val_loss_df.to_csv("linear_svm_val_loss.csv", index=False)

        linear_best_eval_idx = linear_eval_idx[
            np.argmin(val_loss_hist[linear_eval_idx])
        ]
        linear_best_val_loss = np.nanmin(val_loss_hist)
        print(
            "Best validation loss: {:.6f} at epoch {}".format(
                float(linear_best_val_loss), int(linear_best_eval_idx + 1)
            )
        )

        linear_hyperparam_text = (
            f"alpha={linear_alpha}, lambda={linear_lambda}, "
            f"batch_size={linear_batch_size}"
        )
        plt.text(
            0.02,
            0.98,
            linear_hyperparam_text,
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"
            ),
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig("linear_svm_val_loss.png", dpi=150)
        plt.close()

    # kernel SVM stochastic subgradient descent training
    if run_kernel_svm:
        # hyperparameters can be tuned for better validation performance
        kernel_alpha = 0.05
        kernel_lambda = 1e-4
        kernel_batch_size = 32
        kernel_gamma = 1.5
        kernel_num_iter = 30000

        print(
            "[Kernel SVM] Training with alpha={}, lambda_reg={}, batch_size={}, gamma={}".format(
                kernel_alpha, kernel_lambda, kernel_batch_size, kernel_gamma
            )
        )
        kernel_theta_hist, kernel_loss_hist, kernel_val_loss_hist = (
            kernel_svm_subgrad_descent(
                X_train_vect,
                y_train,
                alpha=kernel_alpha,
                lambda_reg=kernel_lambda,
                num_iter=kernel_num_iter,
                batch_size=kernel_batch_size,
                gamma=kernel_gamma,
                X_val=X_val_vect,
                y_val=y_val,
                eval_every=200,
            )
        )

        kernel_theta = kernel_theta_hist[-1]
        kernel_alpha_vec = kernel_theta[:-1]
        kernel_b = kernel_theta[-1]

        # compute accuracy, F1-score, and confusion matrix of kernel SVM on validation set
        K_val_train = gaussian_kernel_matrix(
            X_val_vect, X_train_vect, gamma=kernel_gamma
        )
        kernel_scores_val = K_val_train @ (kernel_alpha_vec * y_train) + kernel_b
        y_val_pred_kernel = np.where(kernel_scores_val >= 0, 1, -1)

        kernel_accuracy = accuracy_score(y_val, y_val_pred_kernel)
        kernel_f1 = f1_score(y_val, y_val_pred_kernel, pos_label=1)
        kernel_cm = confusion_matrix(y_val, y_val_pred_kernel, labels=[-1, 1])

        print("[Kernel SVM]")
        print("Validation Accuracy: {:.4f}".format(kernel_accuracy))
        print("Validation F1-Score: {:.4f}".format(kernel_f1))
        print("Confusion Matrix [[TN, FP], [FN, TP]]:")
        print(kernel_cm)

        # plot kernel SVM validation loss curve
        kernel_eval_idx = np.where(~np.isnan(kernel_val_loss_hist))[0]
        kernel_curve_df = pd.DataFrame(
            {
                "epoch": kernel_eval_idx + 1,
                "val_loss": kernel_val_loss_hist[kernel_eval_idx],
                "alpha": kernel_alpha,
                "lambda_reg": kernel_lambda,
                "batch_size": kernel_batch_size,
                "gamma": kernel_gamma,
            }
        )
        kernel_curve_df.to_csv("kernel_svm_val_loss.csv", index=False)

        best_eval_idx = kernel_eval_idx[
            np.argmin(kernel_val_loss_hist[kernel_eval_idx])
        ]
        best_val_loss = np.nanmin(kernel_val_loss_hist)
        print(
            "Best validation loss: {:.6f} at epoch {}".format(
                float(best_val_loss), int(best_eval_idx + 1)
            )
        )

        plt.figure(figsize=(8, 4))
        plt.plot(
            kernel_curve_df["epoch"],
            kernel_curve_df["val_loss"],
            marker="o",
            markersize=2,
            label="Validation Loss",
        )
        plt.xlabel("Iteration")
        plt.ylabel("Validation Loss")
        plt.title("Kernel SVM Validation Loss")
        hyperparam_text = (
            f"alpha={kernel_alpha}, lambda={kernel_lambda}, "
            f"batch_size={kernel_batch_size}, gamma={kernel_gamma}"
        )
        plt.text(
            0.02,
            0.98,
            hyperparam_text,
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"
            ),
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig("kernel_svm_val_loss.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
