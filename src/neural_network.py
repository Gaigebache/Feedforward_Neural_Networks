"""Core feedforward network and backpropagation, shared by all three models."""

import numpy as np

from activations import activation_derivative, apply_activation
from losses import compute_loss


def initialize_layer(n_inputs, n_outputs, activation, rng):
    """Create one fully connected layer with Glorot (Xavier) uniform initial weights."""
    limit = np.sqrt(6.0 / (n_inputs + n_outputs))
    if activation == "sigmoid":
        limit *= 4.0  # Glorot & Bengio's recommended scale for the logistic function
    return {
        "W": rng.uniform(-limit, limit, size=(n_inputs, n_outputs)),
        "b": np.zeros(n_outputs),
        "activation": activation,
        "trainable": True,
    }


def build_network(layer_sizes, hidden_activation, output_activation, rng):
    """Build a feedforward network from a list of layer sizes."""
    network = []
    for i in range(len(layer_sizes) - 1):
        is_output = i == len(layer_sizes) - 2
        activation = output_activation if is_output else hidden_activation
        network.append(initialize_layer(layer_sizes[i], layer_sizes[i + 1], activation, rng))
    return network


def forward_pass(network, X):
    """Propagate a batch forward. Returns [X, a_1, ..., a_L], the output of every layer."""
    activations = [X]
    for layer in network:
        z = activations[-1] @ layer["W"] + layer["b"]
        activations.append(apply_activation(z, layer["activation"]))
    return activations


def compute_output_delta(outputs, targets, output_activation):
    """Error term dLoss/dz at the output layer (z = the output layer's weighted sum)."""
    n_samples, n_outputs = targets.shape
    if output_activation == "softmax":
        return (outputs - targets) / n_samples
    if output_activation == "linear":
        return 2.0 * (outputs - targets) / (n_samples * n_outputs)
    raise ValueError(f"Output layer must be 'softmax' or 'linear', got '{output_activation}'")


def backward_pass(network, activations, targets):
    """Backpropagation: compute the loss gradient for every layer's weights and biases."""
    delta = compute_output_delta(activations[-1], targets, network[-1]["activation"])
    gradients = [None] * len(network)
    for i in range(len(network) - 1, -1, -1):
        layer_input = activations[i]
        gradients[i] = (layer_input.T @ delta, delta.sum(axis=0))
        if i > 0:
            previous_activation = network[i - 1]["activation"]
            delta = (delta @ network[i]["W"].T) * activation_derivative(layer_input,
                                                                        previous_activation)
    return gradients


def initialize_velocities(network):
    """Zero momentum 'velocity' terms, one (vW, vb) pair per layer."""
    return [(np.zeros_like(layer["W"]), np.zeros_like(layer["b"])) for layer in network]


def update_weights(network, gradients, velocities, learning_rate, momentum):
    """Gradient-descent step with momentum, done in place."""
    new_velocities = []
    for layer, (dW, db), (vW, vb) in zip(network, gradients, velocities):
        if layer["trainable"]:
            vW = momentum * vW - learning_rate * dW
            vb = momentum * vb - learning_rate * db
            layer["W"] += vW
            layer["b"] += vb
        new_velocities.append((vW, vb))
    return new_velocities


def create_minibatches(n_samples, batch_size, rng):
    """Shuffle the row indices and cut them into mini-batches."""
    order = rng.permutation(n_samples)
    return [order[start:start + batch_size] for start in range(0, n_samples, batch_size)]


def copy_network(network):
    """Deep copy of a network, so saved 'best' weights are not changed by later training."""
    return [{"W": layer["W"].copy(), "b": layer["b"].copy(),
             "activation": layer["activation"], "trainable": layer["trainable"]}
            for layer in network]


def predict_network(network, X):
    """Network output for X: class probabilities (softmax) or real values (linear)."""
    return forward_pass(network, X)[-1]


def train_network(network, X, Y, hyperparameters, rng, X_val=None, Y_val=None):
    """Train a network with mini-batch backpropagation (gradient descent + momentum)."""
    output_activation = network[-1]["activation"]
    velocities = initialize_velocities(network)
    history = {"train_loss": [], "val_loss": [], "best_epoch": 0, "diverged": False}
    best_network, best_loss, epochs_without_improvement = copy_network(network), np.inf, 0

    for epoch in range(hyperparameters["max_epochs"]):
        for batch in create_minibatches(len(X), hyperparameters["batch_size"], rng):
            activations = forward_pass(network, X[batch])
            gradients = backward_pass(network, activations, Y[batch])
            velocities = update_weights(network, gradients, velocities,
                                        hyperparameters["learning_rate"],
                                        hyperparameters["momentum"])

        train_loss = compute_loss(Y, predict_network(network, X), output_activation)
        history["train_loss"].append(train_loss)
        if not np.isfinite(train_loss):
            history["diverged"] = True
            break

        if X_val is not None:
            val_loss = compute_loss(Y_val, predict_network(network, X_val), output_activation)
            history["val_loss"].append(val_loss)
            monitored = val_loss
        else:
            monitored = train_loss

        if monitored < best_loss - 1e-6:
            best_loss, best_network = monitored, copy_network(network)
            history["best_epoch"] = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if X_val is not None and epochs_without_improvement >= hyperparameters["patience"]:
                break

    return best_network, history


def numerical_gradient_check(network, X, Y, epsilon=1e-5):
    """Compare backprop gradients to finite-difference gradients."""
    output_activation = network[-1]["activation"]
    analytic = backward_pass(network, forward_pass(network, X), Y)
    worst = 0.0
    for layer, (dW, db) in zip(network, analytic):
        for parameter, gradient in ((layer["W"], dW), (layer["b"], db)):
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                loss_plus = compute_loss(Y, forward_pass(network, X)[-1], output_activation)
                parameter[index] = original - epsilon
                loss_minus = compute_loss(Y, forward_pass(network, X)[-1], output_activation)
                parameter[index] = original
                numeric = (loss_plus - loss_minus) / (2 * epsilon)
                denominator = max(abs(numeric) + abs(gradient[index]), 1e-12)
                worst = max(worst, abs(numeric - gradient[index]) / denominator)
    return worst


def output_activation_for_task(task):
    """Softmax output for classification, linear output for regression."""
    return "softmax" if task == "classification" else "linear"


def split_validation(X, Y, fraction, rng):
    """Hold out a random fraction of the training rows as a validation set for early stopping."""
    order = rng.permutation(len(X))
    n_val = max(1, int(round(fraction * len(X))))
    val, train = order[:n_val], order[n_val:]
    return X[train], Y[train], X[val], Y[val]
