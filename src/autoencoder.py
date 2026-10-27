"""Model 3: feedforward network whose first layer is pre-trained as an autoencoder."""

import numpy as np
from neural_network import build_network, output_activation_for_task, copy_network, initialize_layer, ouput_activation_for_task, predict_network, split_validation, train_network

def build_autoencoder_model(n_inputs, n_hidden, hidden_activation, rng):
    """Input -> Encoding -> Output, where the output has 1 node per input"""
    if n_hidden >= n_inputs:
        raise ValueError(f"Autoencoder must have fewer encoding nodes than inputs, got {n_hidden} >= {n_inputs}")
    return build_network([n_inputs, n_hidden, n_inputs], hidden_activation, "linear", rng)

def train_autoecoder(X, n_encoding, hidden_activation, hyperparameters, rng, X_val=None):
    """Train the autoencoder so its output matches it input (the targets are X itself)."""
    autoencoder = build_autoencoder(X.shape[1], n_encoding, hidden_activation, rng)
    return train_network(autoencoder, X, X, hyperparameters, rng, X_val, X_val)

def reconstruction_error(autoencoder, X):
    """Mean squared reconstruction error per input value"""
    return float(np.mean((predict_network(autoencoder, X) - X) ** 2))

def mean_baseline_error(X_train, X):
    """Reconstruction error of the trivial 'always output the training mean' decoder"""
    return float(np.mean((X - X_train.mean(axis=0)) ** 2))

def attach_prediction_layers(autoencoder, n_hidden2, n_outputs, task, hidden_activation, rng, freeze_encoder):
    """Drop the decoder, keep the trained encoder, and add Hidden 2 and the prediction layer.
    
    Result: Input -> Encoding -> Hidden 2 -> Prediciton. if freeze_encoder is True the encoder weights are locked; otherwise the whole network is fined-tuned by backprop"""

    encoder = copy_network(autoencoder)[0]
    encoder["trainable"] = not freeze_encoder
    n_encoding = encoder["W"].shape[1]
    hidden2 = initialize_layer(n_encoding, n_hidden2, hidden_activation, rng)
    prediction = initialize_layer(n_hidden2, n_outputs, output_activation_for_task(task), rng)
    return [encoder, hidden2, prediction]

def train_autoencoder_model(X_train, Y_train, task, n_encoding, n_hidden2, hidden_activation, autoencoder_hyperparameters, prediction_hyperparameters, rng, freeze_encoder=False, validation_fraction=0.2):
    """Full Model 3 pipeline: pre-train the autoencoder, attach the prediction part, train it.

    Returns (netwrok, info) where info holds both training histories and the recontruction errors, so the report can show how well the encoder compressed the data."""

    X_fit, Y_fit, X_val, Y_val = split_validation(X_train, Y_train, validation_fraction, rng)
    
    autoencoder, autoencoder_history = train_autoencoder(X_fit, n_encoding, hidden_activation, autoencoder_hyperparameters, rng, X_val)

    network, prediction_history = train_network(network, X_fit, Y_fit, prediction_hyperparameters, rng, X_val, Y_val)

    info={"autoencoder_history": autoencoder_history, "prediction_history": prediction_history, 
          "reconstruction_error": reconstruction_error(autoencoder, X_fit), "mean_baseline_error": mean_baseline_error(X_fit, X_fit),
          "freeze_encoder": freeze_encoder}
    return network, info


