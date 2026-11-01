"""Data splitting helpers shared by the tuner and the 5by2 CV harness"""

import numpy as np

def split_halves(indices, y, task, rng):
    """Split indices into 2 halves, stratified by class for classificaiton"""
    indices = np.asarray(indices)
    if task == "classification":
        half_a, half_b = [], []
        for label in np.unique(y[indices]):
            members = rng.permutation(y[indices] == label)
            first, second = members[0::2], members[1::2]
            if rng.random() < 0.5:
                first, second = second, first
            half_a.extend(indices[first])
            half_b.extend(indices[second])
    else:
        order = rng.permutation(indices)
        half_a, half_b = list(order[0::2]), list(order[1::2])
    return np.array(sorted(half_a)), np.array(sorted(half_b))

def holdout_split(y, task, fraction, seed):
    """Pull out a fixed fraction of the rows as a tuning set, stratified by class for classification"""
    rng = np.random.default_rng([seed, 999])
    all_indices = np.arange(len(y))
    if task == "classification":
        tuning = []
        for label in np.unique(y):
            members = rng.permutation(all_indices[y==label])
            n_tuning = max(2, int(round(fraction * len(members))))
            tuning.extend(members[:n_tuning])
    else:
        order = rng.permutation(all_indices)
        tuning = list(order[:max(4, int(round(fraction * len(y))))])
    tuning = np.array(sorted(tuning))
    experiment = np.setdiff1d(all_indices, tuning)
    return experiment, tuning

def repeated_2fold(indices, y, task, repeats, seed):
    """Repeats repititions of 2 fold cross validation, (for our tests we use 5 folds)"""

    folds = []
    for repatitions in range(repeats):
        rng = np.random.default_rng([seed, repatitions])
        half_a, half_b = split_halves(indices, y, task, rng)
        folds.append((half_a, half_b))
        folds.append((half_b, half_a))
    return folds