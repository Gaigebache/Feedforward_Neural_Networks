"""This is the 5x2"""

import argparse
import json
import os
import time

import numpy as np

from cross_validation import (compare_models_statistically, evaluate_predictions,
                              summarize_fold_metrics)
from data_loader import default_data_dir, get_dataset_registry, load_dataset, prepare_fold_data
from neural_network import predict_network
from results_output import (plot_model_comparison, plot_training_curves, print_dataset_summary,
                            save_results_json, write_significance_csv, write_summary_csv)
from split_validation import repeated_2fold
from tuner import train_with_params

N_REPEATS = 5  # 5 x 2-fold = 10 folds, as the 5x2 cv F test requires


def load_tuning_results(name, tuned_dir):
    """Read the <dataset>_tuned.json file written by tuner.py."""
    path = os.path.join(tuned_dir, f"{name}_tuned.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run tuner.py first "
                                f"(e.g. python tuner.py --datasets {name}).")
    with open(path) as file:
        return json.load(file)


def run_model_on_folds(model_name, params, dataset, folds, seed, dataset_index, model_index):
    """Train and test one model with its tuned config on every fold."""
    fold_results = []
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        fold = prepare_fold_data(dataset, train_indices, test_indices)
        rng = np.random.default_rng([seed, dataset_index, model_index, 1000 + fold_index])
        network, history = train_with_params(model_name, fold, params, rng)
        outputs = predict_network(network, fold["X_test"])
        metrics = evaluate_predictions(dataset["task"], fold["y_test"], outputs,
                                       fold["y_scaler"], fold["n_classes"])
        metrics["epochs"] = history["best_epoch"]
        if "reconstruction_error" in history:
            metrics["reconstruction_error"] = history["reconstruction_error"]
        if fold_index == 0:
            metrics["history"] = history
        fold_results.append(metrics)
    return fold_results


def run_dataset(name, data_dir, tuned_dir, dataset_index):
    """The full 5x2 CV experiment for one dataset. Returns a results dict."""
    tuned = load_tuning_results(name, tuned_dir)
    dataset = load_dataset(name, data_dir)
    seed = tuned["seed"]
    experiment_indices = np.array(tuned["experiment_indices"])
    folds = repeated_2fold(experiment_indices, dataset["y"], dataset["task"], N_REPEATS, seed + 2)

    results = {"dataset": name, "task": dataset["task"],
               "n_samples": int(len(experiment_indices)),
               "n_inputs": int(dataset["X"].shape[1]),
               "n_tuning_rows": len(tuned["tuning_indices"]),
               "encoder_mode": tuned["encoder_mode"], "models": {}}
    fold_results_by_model = {}
    for model_index, (model_name, best) in enumerate(tuned["best"].items()):
        started = time.time()
        print(f"  [{name}] 5x2 CV for {model_name} ...", flush=True)
        fold_results = run_model_on_folds(model_name, best["params"], dataset, folds, seed,
                                          dataset_index, model_index)
        fold_results_by_model[model_name] = fold_results
        results["models"][model_name] = {
            "best_config": best["params"], "tuning_val_loss": best["mean_val_loss"],
            "folds": fold_results, "summary": summarize_fold_metrics(fold_results),
            "seconds": round(time.time() - started, 1)}
    results["comparisons"] = compare_models_statistically(fold_results_by_model)
    return results


def main():
    parser = argparse.ArgumentParser(description="Project 3: 5x2 CV experiment on tuned models")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of datasets to run (default: all six)")
    parser.add_argument("--data-dir", default=None,
                        help="folder with the six data files (default: found automatically)")
    parser.add_argument("--tuned-dir", default="tuning_results",
                        help="where tuner.py saved its *_tuned.json files")
    parser.add_argument("--output-dir", default="results")
    arguments = parser.parse_args()
    data_dir = arguments.data_dir or default_data_dir()

    registry_names = sorted(get_dataset_registry())
    os.makedirs(arguments.output_dir, exist_ok=True)
    all_results = []
    for name in arguments.datasets or registry_names:
        print(f"\nDataset: {name}", flush=True)
        results = run_dataset(name, data_dir, arguments.tuned_dir, registry_names.index(name))
        print_dataset_summary(results)
        save_results_json(results, os.path.join(arguments.output_dir, f"{name}.json"))
        plot_model_comparison(results, os.path.join(arguments.output_dir,
                                                    f"{name}_comparison.pdf"))
        plot_training_curves(results, os.path.join(arguments.output_dir, f"{name}_curves.pdf"))
        all_results.append(results)
    write_summary_csv(all_results, os.path.join(arguments.output_dir, "summary.csv"))
    write_significance_csv(all_results, os.path.join(arguments.output_dir, "significance.csv"))
    print(f"\nSaved results, tables and figures to '{arguments.output_dir}/'")


if __name__ == "__main__":
    main()
