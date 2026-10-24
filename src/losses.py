"""Loss functions: cross-entropy (classification) and mean squared error (regression)."""

import numpy as np


def cross_entropy_loss(targets, probabilities):
    """Mean cross-entropy: -(1/n) * sum over samples and classes of y * ln(p)."""
    return float(-np.mean(np.sum(targets * np.log(probabilities + 1e-12), axis=1)))


def mean_squared_error(targets, predictions):
    """Mean of (y - y_hat)^2 over every sample and every output node."""
    return float(np.mean((np.asarray(targets) - np.asarray(predictions)) ** 2))


def compute_loss(targets, outputs, output_activation):
    """Cross-entropy for a softmax output layer, mean squared error for a linear one."""
    if output_activation == "softmax":
        return cross_entropy_loss(targets, outputs)
    return mean_squared_error(targets, outputs)
