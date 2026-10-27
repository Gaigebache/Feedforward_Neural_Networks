# Feedforward Neural Networks

# 1. Setup & Planning
Set up a shared GitHub repo and split tasks (e.g., one person does the core network/backprop code, the other does preprocessing and the CV harness; split the autoencoder and tuning after that)

Agree on a shared data representation (numpy arrays, with one-hot targets for classification and a single column for regression targets)

Agree on a common interface (e.g., fit(X, y), predict(X), predict_proba(X), get_weights()/set_weights())

Decide on a network structure: list of layers, each with weights, biases, activation, and stored values needed for backprop

Decide who owns each piece and a rough timeline (assigned Oct 19, due Nov 9)

# 2. Data Acquisition & Preprocessing

Get one dataset working end to end before scaling to all six.

Download all 6 datasets (UCI or Canvas) and read each .names file

Remove features that act as unique identifiers:

Breast Cancer: Sample code number

Computer Hardware: vendor and model name (drop ERP as well, since it is the authors' linear regression estimate, not a real feature; PRP is the target)

Handle missing values:

Breast Cancer: ? in Bare Nuclei (impute or drop; document the choice)

Congressional Vote: ? means abstain, so keep it as its own value (e.g., a third one-hot category)

One-hot encode categorical attributes (Car Evaluation features, Congressional Vote, Abalone sex M/F/I); decide and document how you handle ordinal features such as Car Evaluation levels

Encode classification targets as one-hot vectors for the softmax output layer

Standardize or min-max scale numeric features (neural networks need this, unlike trees). Compute scaling statistics on the training data only, then apply them to the test/validation data to avoid leakage

Forest Fires: decide whether to log-transform area (it is heavily skewed); document your choice

Consider scaling regression targets for stable training, and remember to invert the scaling before reporting MSE

Produce a clean, consistent structure for all 6 datasets and record the final input dimension of each (needed for the autoencoder size limit)

# 3. Core Building Blocks
   
Activation functions: logistic (sigmoid) and/or tanh with their derivatives (no ReLU or other nonlinearities)

Softmax function (numerically stable: subtract the max before exponentiating)

Linear output layer for regression

Cross-entropy loss for classification and mean squared error for regression

Weight initialization (small random values; fixed seed)

Forward pass for an arbitrary number of layers

Mini-batch or full-batch/online update loop with a learning rate (momentum is optional; document your choice)

Evaluation metrics: classification accuracy or 0/1 loss (plus cross-entropy), and mean squared error for regression

# 4. Linear Networks (Logistic and Linear Regression)
Logistic regression: inputs feed directly into a softmax output layer, trained with cross-entropy and gradient descent

Linear regression: inputs feed directly into one linear output node, trained with MSE and gradient descent (equivalent to OLS)

Verify the gradients with a numerical gradient check on a tiny dataset

Sanity check on one classification and one regression dataset (loss should decrease steadily; compare against a simple baseline such as the plurality class or the mean)

# 5. Backpropagation and the Two-Hidden-Layer Feedforward Network
Implement the backward pass:

Output layer error (softmax + cross-entropy gives prediction minus target; linear + MSE gives a similar form)

Propagate deltas back through both hidden layers using the activation derivatives

Compute weight and bias gradients and apply the updates

Build the network Input => Hidden 1 => Hidden 2 => Prediction (softmax for classification, linear for regression)

Numerical gradient check on the full two-hidden-layer network

Sanity check: the network should be able to overfit a very small subset of one dataset

Choose a stopping rule (fixed number of epochs, or early stopping on a validation set) and document it

# 6. Autoencoder

Build the autoencoder: Input => Encoding (one hidden layer) => Output (same number of nodes as the input), trained with squared reconstruction error

Make sure the number of encoding nodes is strictly less than the number of inputs (watch Computer Hardware, which has very few features after dropping identifiers)

Use a linear output layer for reconstruction on the scaled inputs, and decide how to handle one-hot inputs (document the choice)

Check that reconstruction error decreases and ends up better than a trivial baseline

Remove the decoder (output) layer and attach the second hidden layer and prediction layer to the encoder: Input => Encoding => Hidden 2 => Prediction

Decide between the two training options and explain the choice in the report:

Lock the encoder weights and train only the new layers, or Fine-tune the whole network with backprop

Confirm the encoder weights are copied correctly (and truly frozen, if you chose that option)

Train the autoencoder on training data only (never on test data)

# 7. Hyperparameter Tuning
Decide on a tuning protocol that keeps the test folds untouched (e.g., hold out a validation set, stratified for classification, and tune on it before running 5x2 CV; or tune inside the training fold only)

Choose the hyperparameters to tune and the grid for each:

Hidden nodes in layer 1 and layer 2 (feedforward network)

Encoding nodes (autoencoder network; strictly fewer than the inputs) and hidden nodes in layer 2

Learning rate, momentum (if used), and number of epochs

Learning rate and epochs for the linear networks

Keep the search space manageable (large datasets like Abalone are slower; consider tuning on a subset of the grid)

Record the tuning results to files (all configurations and scores), not just the winner

Record the final chosen hyperparameters per dataset and per model for the report

# 8. 5x2 Cross-Validation Harness

Implement 5x2 CV: 5 repetitions of 2-fold CV with shuffling, stratified for classification

For each fold, train all three models on the training half and test on the other half:

Linear network (logistic or linear regression)

Feedforward network with two hidden layers

Autoencoder-based network

Fit the preprocessing (scaling, imputation) inside each training fold to avoid leakage

Use the same folds for all models so results are paired

Run all models across all 6 datasets (3 models x 6 datasets x 10 runs = 180 trained networks)

Save raw predictions, scores, loss curves, and chosen hyperparameters to files rather than only printing them

Set random seeds and document them for reproducibility

Estimate the total runtime early and cut the epoch or grid sizes if it is too slow

# 9. Results Analysis

Aggregate results into tables (mean +/- std for accuracy/0-1 loss or MSE per dataset and model)

Statistical comparison of the models (e.g., paired t-test, Wilcoxon signed-rank, or the 5x2 cv F-test), and state your significance level

Figures such as performance by model per dataset, boxplots of fold scores, and training loss curves for the three networks

Figure or table showing the autoencoder reconstruction error and how it affects downstream performance

Compare the autoencoder network against the plain feedforward network (with the encoder locked vs. fine-tuned, if you try both)

Compare the feedforward networks against the linear networks (does the extra depth help?)

For Computer Hardware, compare against the authors' linear regression estimates as the project suggests

Note where each method helped, hurt, or made no difference, and think about why (dataset size, number of features, noise, class balance)

# 10. Report Writing (JMLR format, <=15 pages + appendices)

Title and authors

Problem statement with at least one testable hypothesis (e.g., "The autoencoder-pretrained network will outperform the standard two-hidden-layer network on datasets with many input features, and both will outperform the linear networks on datasets with nonlinear structure")

Experimental approach and project design (datasets, preprocessing choices, tuning protocol, 5x2 CV, locked vs. fine-tuned encoder choice)

Algorithm explanations without code (logistic and linear regression networks, backpropagation, softmax with cross-entropy, autoencoder and how it is attached to the prediction layers)

Results (tables and figures with explanations)

Analysis and discussion, tied back to your hypothesis

Conclusion

References (if using any beyond the course content)

Appendix A: main lessons learned (not counted in the page limit)

Appendix B: who did what (not counted in the page limit)

# 11. Final Checks
All math is typeset in LaTeX (including softmax, cross-entropy, MSE, the autoencoder loss, and the backprop updates)

Figures are vector or high quality, with no screenshots

Proofread against JMLR format requirements

Confirm the 15-page limit (excluding appendices)
[ ] Export to PDF and check plagiarism-safe wording (the report goes through TurnItIn)
[ ] Submit the PDF to the portal named in the assignment ("P2 Paper" in the spec text; confirm the exact name on Canvas) before the Nov 9 deadline
