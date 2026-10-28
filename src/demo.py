import numpy as np
import os
from data_loader import *
from linear_model import *
from neural_network import *
from autoencoder import *

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")

HYPER = {"learning_rate": 0.01, "momentum": 0.9, "batch_size": 32,
         "max_epochs": 300, "patience": 20}


def demo(name, data_dir=DATA_DIR):
    rng = np.random.default_rng(0)
    dataset = load_dataset(name, data_dir)
    task, n_rows = dataset["task"], len(dataset["y"])
    print(f"\n=== {name}: {n_rows} rows, {dataset['X'].shape[1]} inputs, {task} ===")

    order = rng.permutation(n_rows)                       # one simple 50/50 split, no CV yet
    fold = prepare_fold_data(dataset, order[: n_rows // 2], order[n_rows // 2:])
    X, Y = fold["X_train"], fold["Y_train"]
    X_fit, Y_fit, X_val, Y_val = split_validation(X, Y, 0.2, rng)
    n_in, n_out = X.shape[1], Y.shape[1]

    models = {}
    models["linear"] = train_network(build_linear_model(n_in, n_out, task, rng),
                                     X_fit, Y_fit, HYPER, rng, X_val, Y_val)
    models["feedforward"] = train_network(
        build_network([n_in, 16, 8, n_out], "tanh", output_activation_for_task(task), rng),
        X_fit, Y_fit, HYPER, rng, X_val, Y_val)
    network, info = train_autoencoder_model(X, Y, task, min(16, n_in - 1), 8, "tanh",
                                            HYPER, HYPER, rng)
    models["autoencoder"] = (network, info["prediction_history"])
    print(f"autoencoder reconstruction error {info['reconstruction_error']:.3f} "
          f"vs mean-only baseline {info['mean_baseline_error']:.3f}")

    for model_name, (net, history) in models.items():
        outputs = predict_network(net, fold["X_test"])
        if task == "classification":
            score = np.mean(np.argmax(outputs, axis=1) == fold["y_test"])
            baseline = np.bincount(dataset["y"]).max() / n_rows
            result = f"test accuracy {score:.3f} (plurality baseline {baseline:.3f})"
        else:
            preds = invert_standardizer(outputs.ravel(), fold["y_scaler"])
            mse = np.mean((fold["y_test"] - preds) ** 2)
            baseline = np.mean((fold["y_test"] - fold["y_scaler"]["mean"]) ** 2)
            result = f"test MSE {mse:.3f} (predict-the-mean baseline {baseline:.3f})"
        print(f"{model_name:12s} train loss {history['train_loss'][0]:.3f} -> "
              f"{history['train_loss'][-1]:.3f}, best epoch {history['best_epoch']}, {result}")


demo("congressional_votes")
demo("computer_hardware")
demo("abalone")
demo("car")
demo("forest_fires")
demo("breast_cancer")