"""Loading and preprocessing the six UCI datasets."""

import numpy as np
import os
from collections import Counter


def read_delimited_file(path, delimiter=",", skip_header=False):
    """Read a text data file into a list of rows, where each row is a list of strings."""

    rows = []
    with open(path, "r") as file:
        for line_number, line in enumerate(file):
            if skip_header and line_number == 0:
                continue
            line = line.strip()
            if not line:
                continue
            rows.append([value.strip() for value in line.split(delimiter)])
    return rows


def impute_missing_with_mode(column_values, missing_marker="?"):
    present = [value for value in column_values if value != missing_marker]
    mode = Counter(present).most_common(1)[0][0]
    return [mode if value == missing_marker else value for value in column_values]


def one_hot_encode(column_values, categories=None):
    """Turn a column of categorical strings into a 0/1 matrix with one column per category."""
    if categories is None:
        categories = sorted(set(column_values))
    index_of = {category: i for i, category in enumerate(categories)}
    matrix = np.zeros((len(column_values), len(categories)))
    for row, value in enumerate(column_values):
        matrix[row, index_of[value]] = 1.0
    return matrix, list(categories)


def encode_class_labels(labels, class_names=None):
    """Map class label strings to integers 0..k-1."""
    if class_names is None:
        class_names = sorted(set(labels))
    index_of = {name: i for i, name in enumerate(class_names)}
    return np.array([index_of[label] for label in labels], dtype=int), list(class_names)


def load_breast_cancer(path):
    """Load breast-cancer-wisconsin.data."""
    rows = read_delimited_file(path)
    columns = list(zip(*rows))
    features = []
    for column in columns[1:10]:
        features.append([float(v) for v in impute_missing_with_mode(list(column))])
    X = np.array(features).T
    names = {"2": "benign", "4": "malignant"}
    y, class_names = encode_class_labels([names[v] for v in columns[10]])
    return {"name": "breast_cancer", "task": "classification", "X": X, "y": y,
            "class_names": class_names}


def load_car(path):
    """Load car.data."""
    rows = read_delimited_file(path)
    columns = list(zip(*rows))
    categories = [
        ["vhigh", "high", "med", "low"],      # buying
        ["vhigh", "high", "med", "low"],      # maint
        ["2", "3", "4", "5more"],             # doors
        ["2", "4", "more"],                   # persons
        ["small", "med", "big"],              # lug_boot
        ["low", "med", "high"],               # safety
    ]
    blocks = [one_hot_encode(list(columns[i]), categories[i])[0] for i in range(6)]
    X = np.hstack(blocks)
    y, class_names = encode_class_labels(list(columns[6]), ["unacc", "acc", "good", "vgood"])
    return {"name": "car", "task": "classification", "X": X, "y": y,
            "class_names": class_names}


def load_congressional_votes(path):
    """Load the congressional voting records."""
    rows = read_delimited_file(path)
    rows = [[value.strip('"') for value in row] for row in rows]
    if rows[0][0].lower() == "class":
        rows = rows[1:]
    party_names = {"1": "republican", "0": "democrat",
                   "republican": "republican", "democrat": "democrat"}
    vote_value = {"y": 1.0, "n": -1.0, "?": 0.0, "NA": 0.0}
    X = np.array([[vote_value[v] for v in row[1:17]] for row in rows])
    y, class_names = encode_class_labels([party_names[row[0]] for row in rows])
    return {"name": "congressional_votes", "task": "classification", "X": X, "y": y,
            "class_names": class_names}


def load_abalone(path):
    """Load abalone.data."""
    rows = read_delimited_file(path)
    sex, _ = one_hot_encode([row[0] for row in rows], ["M", "F", "I"])
    measurements = np.array([[float(v) for v in row[1:8]] for row in rows])
    X = np.hstack([sex, measurements])
    y = np.array([float(row[8]) for row in rows])
    return {"name": "abalone", "task": "regression", "X": X, "y": y, "class_names": None}


def load_computer_hardware(path):
    """Load machine.data."""
    rows = read_delimited_file(path)
    X = np.array([[float(v) for v in row[2:8]] for row in rows])
    y = np.array([float(row[8]) for row in rows])
    return {"name": "computer_hardware", "task": "regression", "X": X, "y": y,
            "class_names": None}


def load_forest_fires(path, log_target=True):
    """Load forestfires.csv (has a header row)."""
    rows = read_delimited_file(path, skip_header=True)
    months = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]
    days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    coordinates = np.array([[float(row[0]), float(row[1])] for row in rows])
    month, _ = one_hot_encode([row[2] for row in rows], months)
    day, _ = one_hot_encode([row[3] for row in rows], days)
    weather = np.array([[float(v) for v in row[4:12]] for row in rows])
    X = np.hstack([coordinates, month, day, weather])
    area = np.array([float(row[12]) for row in rows])
    y = np.log1p(area) if log_target else area
    return {"name": "forest_fires", "task": "regression", "X": X, "y": y, "class_names": None}


def get_dataset_registry():
    """Return {dataset name: (loader function, accepted file names in the data folder)}."""
    return {
        "breast_cancer": (load_breast_cancer,
                          ["breast-cancer-wisconsin.data", "breast-cancer.data"]),
        "car": (load_car, ["car.data"]),
        "congressional_votes": (load_congressional_votes,
                                ["house-votes-84.data", "HouseVotes84.csv"]),
        "abalone": (load_abalone, ["abalone.data"]),
        "computer_hardware": (load_computer_hardware, ["machine.data"]),
        "forest_fires": (load_forest_fires, ["forestfires.csv"]),
    }


def load_dataset(name, data_dir):
    """Load one of the six project datasets by name from data_dir."""
    registry = get_dataset_registry()
    if name not in registry:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {sorted(registry)}")
    loader, file_names = registry[name]
    for file_name in file_names:
        path = os.path.join(data_dir, file_name)
        if os.path.exists(path):
            return loader(path)
    raise FileNotFoundError(
        f"No file for '{name}' in {data_dir}/ (expected one of {file_names}). "
        "Run download_datasets.py or download it from UCI.")


def fit_standardizer(values):
    """Compute the per-column mean and standard deviation of the TRAINING data."""
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return {"mean": mean, "std": std}


def apply_standardizer(values, standardizer):
    """Z-score values using a mean/std fitted on the training data."""
    return (values - standardizer["mean"]) / standardizer["std"]


def invert_standardizer(values, standardizer):
    """Undo apply_standardizer, mapping z-scores back to the original units."""
    return values * standardizer["std"] + standardizer["mean"]


def one_hot_targets(labels, n_classes):
    """Turn integer class labels into one-hot target rows for softmax / cross-entropy."""
    targets = np.zeros((len(labels), n_classes))
    targets[np.arange(len(labels)), labels] = 1.0
    return targets


def prepare_fold_data(dataset, train_indices, test_indices):
    """Build network-ready train/test arrays for one fold."""
    X, y = dataset["X"], dataset["y"]
    x_scaler = fit_standardizer(X[train_indices])
    fold = {
        "X_train": apply_standardizer(X[train_indices], x_scaler),
        "X_test": apply_standardizer(X[test_indices], x_scaler),
        "y_test": y[test_indices],
        "y_scaler": None,
        "n_classes": None,
    }
    if dataset["task"] == "classification":
        n_classes = len(dataset["class_names"])
        fold["n_classes"] = n_classes
        fold["Y_train"] = one_hot_targets(y[train_indices], n_classes)
    else:
        y_scaler = fit_standardizer(y[train_indices])
        fold["y_scaler"] = y_scaler
        fold["Y_train"] = apply_standardizer(y[train_indices], y_scaler).reshape(-1, 1)
    return fold
