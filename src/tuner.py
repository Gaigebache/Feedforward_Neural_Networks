""" Hyperparameter tuning by random search, done on a held-out tuning set.
Protocol:
1. Pull out fixed tuning set, 20% of the rows by default, stratified for classification tasks.
2. For each model, sample random hyperparameter configs. Each config is scored with repeated 2 fold cross-validation
inside the tuning set
3. The config with the lowest mean validation loss wins (cross entropy for classification, MSE on regression)
4. Number of epochs is not searched, we stop training by useing early stopping based on a validation split carved out 
of each training fold, w/ max_epochs as un upper bound
"""

import argparse
import numpy as np
import csv
import json
import time

from autoencoder import *
from data_loader import *
from linear_model import *
from neural_network import *
from losses import *
#from split_validation import *

MODEL_NAMES = ["linear", "feedforward", "autoencoder"]

ACTIVATIONS = ["sigmoid", "tanh"]
HIDDEN1_SIZES = [4, 8, 16, 32]
HIDDEN2_SIZES = [4, 8, 16]
ENCODING_FRACTIONS = [0.25, 0.5, 0.75]
MOMENTUMS = [0.0, 0.5, 0.9]
BATCH_SIZES = [16,32,64]
LEARNING_RATES_RANGE = [0.001, 0.1]

FIXED = {"max_epochs": 500, "patience": 20, "validation_fraction": 0.2}

def sample_hyperparameters(model_name, n_inputs, rng, encoder_mode):
    low, high = LEARNING_RATES_RANGE
    def log_uniform():
        return round(np.exp(rng.uniform(np.log(low), np.log(high))), 5)

    params = {"learning_rate": log_uniform(),
              "momentum": float(rng.choice(MOMENTUMS)),
              "batch_size": int(rng.choice(BATCH_SIZES))}

    if model_name == "linear":
        return params

    params["hidden_activation"] = str(rng.choice(ACTIVATIONS))
    params["n_hidden2"] = int(rng.choice(HIDDEN2_SIZES))
    if model_name == "feedforward":
        params["n_hidden1"] = int(rng.choice(HIDDEN1_SIZES))
        return params

    #Autoencoder model: the encoding layer must have strictly fewer nodes than the inputs.
    encoding_sizes = sorted({max(1, min(n_inputs - 1, int(round(f * n_inputs)))) for f in ENCODING_FRACTIONS})
    params["n_encoding"] = int(rng.choice(encoding_sizes))
    params["ae_learning_rate"] = log_uniform()
    if encoder_mode == "tuned":
        params["freeze_encoder"] = bool(rng.random() < 0.5)
    else:
        params["freeze_encoder"] = encoder_mode == "frozen"
    return params

def train_with_params(model_name, fold, params, rng):
    """Train one model w/ one config on one fold's training data."""
    X, Y = fold["X_train"], fold["Y_train"]
    task = "classification" if fold["n_classes"] is not None else "regression"
    hyperparams = {"learning_rate": params["learning_rate"],
                  "momentum": params["momentum"],
                  "batch_size": params["batch_size"],
                  "max_epochs": FIXED["max_epochs"],
                  "patience": FIXED["patience"]}
    
    if model_name in ("linear", "feedforward"):
        X_fit, Y_fit, X_val, Y_val = split_validation(X, Y, FIXED["validation_fraction"], rng)
        if model_name == "linear":
            network = build_linear_model(X.shape[1], Y.shape[1], task, rng)
        else:
            sizes = [X.shape[1], params["n_hidden1"], params["n_hidden2"], Y.shape[1]]
            network = build_feedforward_network(sizes, params["hidden_activation"], rng)
        network, _ = train_network(network, X_fit, Y_fit, hyperparams, rng, X_val, Y_val)
        return network

    pretraining = dict(hyperparams, learning_rate = params["ae_learning_rate"])
    network, _ = train_autoencoder_model(X, Y, task, params["n_encoding"], params["n_hidden2"], params["hidden_activation"],
                                          pretraining, hyperparams, rng, freeze_encoder = params["freeze_encoder"], 
                                          validation_fraction=FIXED["validation_fraction"])
    return network

def evaluate_fold(network, fold):
    """Return on the folds held out half"""

    outputs = predict_network(network, fold["X_test"])
    if fold["n_classes"] is not None:
        targets = one_hot_targets(fold["Y_test"], fold["n_classes"])
        accuracy = float(np.mean(np.argmax(outputs, axis=1) == fold["Y_test"]))
        return cross_entropy_loss(targets, outputs), accuracy
    scaled_targets = apply_standardizer(fold["Y_test"], fold["Y_scaler"]).reshape(-1, 1)
    orignal_predictions = invert_standardizer(outputs.ravel(), fold["Y_scaler"])
    return (mean_squared_error(scaled_targets, outputs), mean_squared_error(fold["Y_test"], orignal_predictions))

def tune_dataset(dataset, dataset_index, models, trials, repeats, tuning_fraction, seed,
                 encoder_mode, output_dir):
    """Random-search every requested model on one dataset and write the log and best configs."""
    name, task, y = dataset["name"], dataset["task"], dataset["y"]
    n_inputs = dataset["X"].shape[1]
    experiment_indices, tuning_indices = holdout_split(y, task, tuning_fraction, seed)
    folds = repeated_2fold(tuning_indices, y, task, repeats, seed + 1)
    prepared = [prepare_fold_data(dataset, train, test) for train, test in folds]
    print(f"\n=== {name}: {len(tuning_indices)} tuning rows, {len(experiment_indices)} "
          f"experiment rows, {n_inputs} inputs, {len(prepared)} inner folds ===", flush=True)
 
    log_rows, best_configs = [], {}
    for model_index, model_name in enumerate(models):
        sampler = np.random.default_rng([seed, dataset_index, model_index])
        best = None
        for trial in range(trials):
            params = sample_params(model_name, n_inputs, sampler, encoder_mode)
            losses, scores = [], []
            for fold_index, fold in enumerate(prepared):
                rng = np.random.default_rng([seed, dataset_index, model_index, trial, fold_index])
                network = train_with_params(model_name, fold, params, rng)
                loss, score = evaluate_fold(network, fold)
                losses.append(loss if np.isfinite(loss) else np.inf)
                scores.append(score)
            mean_loss = float(np.mean(losses))
            log_rows.append({"dataset": name, "model": model_name, "trial": trial,
                             "mean_val_loss": mean_loss,
                             "std_val_loss": float(np.std(losses)),
                             "mean_score": float(np.mean(scores)),
                             "params": json.dumps(params)})
            if best is None or mean_loss < best["mean_val_loss"]:
                best = {"mean_val_loss": mean_loss, "mean_score": float(np.mean(scores)),
                        "trial": trial, "params": params}
            print(f"{name:20s} {model_name:12s} trial {trial + 1:3d}/{trials}  "
                  f"val loss {mean_loss:.4f}  best so far {best['mean_val_loss']:.4f}",
                  flush=True)
        best_configs[model_name] = best
 
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{name}_tuning_log.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    with open(os.path.join(output_dir, f"{name}_tuned.json"), "w") as file:
        json.dump({"dataset": name, "task": task, "seed": seed, "trials": trials,
                   "inner_repeats": repeats, "tuning_fraction": tuning_fraction,
                   "encoder_mode": encoder_mode, "fixed": FIXED, "best": best_configs,
                   "experiment_indices": experiment_indices.tolist(),
                   "tuning_indices": tuning_indices.tolist()}, file, indent=2)
    return best_configs

def main():
    default_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")
    parser = argparse.ArgumentParser(description="Project 3: random-search hyperparameter tuning")
    parser.add_argument("--data-dir", default=default_data)
    parser.add_argument("--output-dir", default="tuning_results")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of datasets to tune (default: all six)")
    parser.add_argument("--models", nargs="*", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--trials", type=int, default=30,
                        help="Random configurations tried per model per dataset")
    parser.add_argument("--repeats", type=int, default=2,
                        help="Repetitions of 2-fold CV inside the tuning set")
    parser.add_argument("--tuning-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=447)
    parser.add_argument("--encoder", choices=["finetune", "frozen", "tune"], default="finetune",
                        help="Autoencoder model: fine-tune the encoder, freeze it, or let the "
                             "tuner choose")
    arguments = parser.parse_args()
 
    registry_names = sorted(get_dataset_registry())
    for name in arguments.datasets or registry_names:
        started = time.time()
        dataset = load_dataset(name, arguments.data_dir)
        best = tune_dataset(dataset, registry_names.index(name), arguments.models,
                            arguments.trials, arguments.repeats, arguments.tuning_fraction,
                            arguments.seed, arguments.encoder, arguments.output_dir)
        print(f"--- {name} done in {time.time() - started:.0f}s ---")
        for model_name, result in best.items():
            print(f"  {model_name:12s} best val loss {result['mean_val_loss']:.4f}  "
                  f"{result['params']}")
 
 
if __name__ == "__main__":
    main()


#TODO:5x2 cross validation so we can actually run the experiments and tuner.py