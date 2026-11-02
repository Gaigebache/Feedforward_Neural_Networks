"""Model 2: feedforward network with two hidden layers."""

from neural_network import build_network, output_activation_for_task


def build_feedforward_model(n_inputs, n_outputs, task, hidden_1, hidden_2, activation, rng):
    """Input -> Hidden 1 -> Hidden 2 -> Prediction (Figure 2 of the assignment)."""
    return build_network([n_inputs, hidden_1, hidden_2, n_outputs], activation,
                         output_activation_for_task(task), rng)
