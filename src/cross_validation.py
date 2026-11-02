"""Evaluation metrics and the 5x2 cv F test"""

from itertools import combinations

import numpy as np
from scipy.stats import f as f_distribution

from data_loader import invert_standardizer
from losses import mean_squared_error


def accuracy_score(true_labels, predicted_labels):
    """Fraction of predictions that match the true class."""
    return float(np.mean(np.asarray(true_labels) == np.asarray(predicted_labels)))


def macro_f1_score(true_labels, predicted_labels, n_classes):
    """Unweighted mean of the per-class F1 scores."""
    true_labels, predicted_labels = np.asarray(true_labels), np.asarray(predicted_labels)
    scores = []
    for c in range(n_classes):
        true_positive = np.sum((predicted_labels == c) & (true_labels == c))
        predicted = np.sum(predicted_labels == c)
        actual = np.sum(true_labels == c)
        if predicted == 0 and actual == 0:
            continue
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        total = precision + recall
        scores.append(2 * precision * recall / total if total else 0.0)
    return float(np.mean(scores))


def r_squared_score(true_values, predicted_values):
    """Coefficient of determination: 1 - SSE / SST (1 is perfect, 0 = predicting the mean)."""
    true_values, predicted_values = np.asarray(true_values), np.asarray(predicted_values)
    residual = np.sum((true_values - predicted_values) ** 2)
    total = np.sum((true_values - true_values.mean()) ** 2)
    return float(1.0 - residual / total) if total > 0 else 0.0


def evaluate_predictions(task, true_values, network_outputs, y_scaler=None, n_classes=None):
    """Score one fold's test predictions."""
    if task == "classification":
        predicted = np.argmax(network_outputs, axis=1)
        accuracy = accuracy_score(true_values, predicted)
        return {"loss": 1.0 - accuracy, "accuracy": accuracy,
                "macro_f1": macro_f1_score(true_values, predicted, n_classes)}
    predicted = invert_standardizer(network_outputs.ravel(), y_scaler)
    mse = mean_squared_error(true_values, predicted)
    return {"loss": mse, "mse": mse, "rmse": float(np.sqrt(mse)),
            "r2": r_squared_score(true_values, predicted)}


def five_by_two_cv_f_test(losses_a, losses_b):
    """Alpaydin's combined 5x2 cv F test for whether two models perform differently."""
    differences = (np.asarray(losses_a) - np.asarray(losses_b)).reshape(5, 2)
    variances = ((differences - differences.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    denominator = 2.0 * variances.sum()
    if denominator == 0:
        return (0.0, 1.0) if np.all(differences == 0) else (float("inf"), 0.0)
    statistic = float(np.sum(differences ** 2) / denominator)
    return statistic, float(f_distribution.sf(statistic, 10, 5))


def summarize_fold_metrics(fold_results):
    """Mean and standard deviation of every numeric metric across the 10 folds."""
    summary = {}
    for key, value in fold_results[0].items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values = np.array([fold[key] for fold in fold_results], dtype=float)
            summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
                            "median": float(np.median(values))}
    return summary


def compare_models_statistically(fold_results_by_model, alpha=0.05):
    """Run the 5x2 cv F test on every pair of models (all models must use the SAME folds)."""
    comparisons = []
    for model_a, model_b in combinations(fold_results_by_model, 2):
        losses_a = [fold["loss"] for fold in fold_results_by_model[model_a]]
        losses_b = [fold["loss"] for fold in fold_results_by_model[model_b]]
        statistic, p_value = five_by_two_cv_f_test(losses_a, losses_b)
        better = model_a if sum(losses_a) < sum(losses_b) else model_b
        comparisons.append({"model_a": model_a, "model_b": model_b, "F": statistic,
                            "p_value": p_value, "lower_loss": better,
                            "significant": p_value < alpha})
    return comparisons
