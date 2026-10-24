"""Activation functions and their derivatives."""

import numpy as np


def sigmoid(z):
    """Logistic activation 1 / (1 + e^-z). Inputs are clipped to avoid overflow in exp."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(activation):
    """Derivative of the logistic function, written in terms of its OUTPUT a: a(1 - a)."""
    return activation * (1.0 - activation)


def tanh_activation(z):
    """Hyperbolic tangent activation, output in (-1, 1)."""
    return np.tanh(z)


def tanh_derivative(activation):
    """Derivative of tanh, written in terms of its OUTPUT a: 1 - a^2."""
    return 1.0 - activation ** 2


def softmax(z):
    """Row-wise softmax. Subtracting the row max first keeps exp from overflowing."""
    shifted = z - z.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def apply_activation(z, name):
    """Apply the named activation: 'sigmoid', 'tanh', 'softmax' or 'linear'."""
    if name == "sigmoid":
        return sigmoid(z)
    if name == "tanh":
        return tanh_activation(z)
    if name == "softmax":
        return softmax(z)
    if name == "linear":
        return z
    raise ValueError(f"Unknown activation '{name}'")


def activation_derivative(activation, name):
    """Derivative of a HIDDEN-layer activation, given that layer's output values."""
    if name == "sigmoid":
        return sigmoid_derivative(activation)
    if name == "tanh":
        return tanh_derivative(activation)
    raise ValueError(f"Hidden layers must use 'sigmoid' or 'tanh', got '{name}'")
