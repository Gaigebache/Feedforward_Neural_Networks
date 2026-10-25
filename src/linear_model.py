"""Model 1: linear network (logistic regression / least-squares regression)."""

from neural_network import build_network, output_activation_for_task


def build_linear_model(n_inputs, n_outputs, task, rng):
    """Inputs wired straight to the output layer"""
    return build_network([n_inputs, n_outputs], None, output_activation_for_task(task), rng)
