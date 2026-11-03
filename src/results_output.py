
import csv
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def print_dataset_summary(results):
    """Print a readable table of one dataset's results and its significance tests."""
    task = results["task"]
    metrics = ["accuracy", "macro_f1"] if task == "classification" else ["mse", "rmse", "r2"]
    print(f"\n=== {results['dataset']} ({task}, n={results['n_samples']}, "
          f"inputs={results['n_inputs']}) ===")
    print(f"{'model':<13}" + "".join(f"{m:>22}" for m in metrics) + f"{'epochs':>10}")
    for model_type, model in results["models"].items():
        summary = model["summary"]
        cells = "".join(f"{summary[m]['mean']:>12.4f} ± {summary[m]['std']:<7.4f}"
                        for m in metrics)
        print(f"{model_type:<13}{cells}{summary['epochs']['mean']:>10.0f}")
    for model_type, model in results["models"].items():
        print(f"  {model_type} tuned config: {model['best_config']}")
    print("5x2 cv F tests (H0: the two models perform the same):")
    for c in results["comparisons"]:
        verdict = f"significant, {c['lower_loss']} better" if c["significant"] else "not significant"
        print(f"  {c['model_a']} vs {c['model_b']}: F = {c['F']:.3f}, "
              f"p = {c['p_value']:.4f} -> {verdict}")


def convert_to_json_ready(value):
    """Recursively turn numpy arrays / numbers into plain Python types so json can save them."""
    if isinstance(value, dict):
        return {str(k): convert_to_json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [convert_to_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def save_results_json(results, path):
    """Save the complete results (every fold, tuning table and training curve) as JSON."""
    with open(path, "w") as file:
        json.dump(convert_to_json_ready(results), file, indent=2)


def write_summary_csv(all_results, path):
    """One row per (dataset, model): mean and std of each metric plus the tuned config."""
    columns = ["dataset", "task", "model", "metric", "mean", "std", "best_config"]
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for results in all_results:
            for model_type, model in results["models"].items():
                for metric, stats in model["summary"].items():
                    writer.writerow([results["dataset"], results["task"], model_type, metric,
                                     f"{stats['mean']:.6f}", f"{stats['std']:.6f}",
                                     model["best_config"]])


def write_significance_csv(all_results, path):
    """One row per (dataset, model pair) with the 5x2 cv F test result."""
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "model_a", "model_b", "F", "p_value",
                         "lower_loss", "significant_at_0.05"])
        for results in all_results:
            for c in results["comparisons"]:
                writer.writerow([results["dataset"], c["model_a"], c["model_b"],
                                 f"{c['F']:.4f}", f"{c['p_value']:.4f}",
                                 c["lower_loss"], c["significant"]])


def plot_model_comparison(results, path):
    """Box plot of the 10 test-fold scores of each model on one dataset (saved as PDF/PNG)."""
    metric = "accuracy" if results["task"] == "classification" else "mse"
    labels = {"linear": "Linear", "feedforward": "Feedforward\n(2 hidden)",
              "autoencoder": "Autoencoder\n+ hidden"}
    models = list(results["models"])
    data = [[fold[metric] for fold in results["models"][m]["folds"]] for m in models]

    figure, axis = plt.subplots(figsize=(4.5, 3.2))
    axis.boxplot(data, widths=0.5)
    axis.set_xticks(range(1, len(models) + 1), [labels[m] for m in models])
    axis.set_ylabel("Accuracy" if metric == "accuracy" else "MSE")
    axis.set_title(results["dataset"].replace("_", " ").title() + " (5x2 CV)")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_training_curves(results, path):
    """Training and validation loss per epoch on fold 1, one panel per model."""
    models = list(results["models"])
    figure, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3))
    loss_name = "Cross-entropy" if results["task"] == "classification" else "MSE (z-scored target)"
    for axis, model_type in zip(axes, models):
        history = results["models"][model_type]["folds"][0]["history"]
        axis.plot(history["train_loss"], label="train")
        if history["val_loss"]:
            axis.plot(history["val_loss"], label="validation")
        if "autoencoder" in history:
            axis.plot(history["autoencoder"]["val_loss"], "--", label="AE reconstruction")
        axis.axvline(history["best_epoch"] - 1, color="gray", linestyle=":", linewidth=1)
        axis.set_title(model_type)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(loss_name)
        axis.legend(fontsize=8)
    figure.suptitle(results["dataset"].replace("_", " ").title())
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
