'''
Implement logistic regression manually with gradient descent
'''
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import tqdm
import os

############## Helper Functions ##############

def generate_dataset(coeffs, n_samples):
    '''
    Construct a simple dataset

    :param coeffs: array of coefficients -- shape (n_features,)
    :param n_samples: number of samples to generate

    :return X: array of samples -- shape (n_samples, n_features)
    :return y: array of labels -- shape (n_samples,)
    '''
    # Generate samples, where each sample is n_features random numbers
    X = np.random.rand(n_samples, coeffs.shape[0])

    # Assign labels, according to a function f(x) = \sum_i c_i * x_i, for some coeffs c_i
    #   y = 1 if f(x) > 0
    #   y = 0 otherwise
    y = np.dot(X, coeffs) > 0

    return X, y

def sigmoid(x):
    '''
    Sigmoid activation
    '''
    return 1 / (1 + np.exp(-x))

def forward_pass(X, w, b):
    '''
    Compute forward pass over all samples, for a given weight vector
    '''
    return sigmoid(np.dot(X,w)+b)

def loss(y_pred, y):
    '''
    Compute loss, averaged over samples
    '''
    return np.sum(-( y*np.log(y_pred) + (1-y)*np.log(1-y_pred) )) / y.shape[0]

def backpropagate(X, w, b, y):
    '''
    Compute gradient of loss wrt inputs: (dL/dw_0, dL/dw1, ...) -- averaged over all samples
    '''
    z = np.dot(X,w) + b

    # Chain rule for dL_dz -- simplifies algebraically
    #dL_dsigma = (sigmoid(z) - y) / (sigmoid(z)*(1-sigmoid(z)))
    #dsigma_dz = sigmoid(z)*(1-sigmoid(z))
    dL_dz = sigmoid(z) - y # (n_samples,)
    dL_dz = dL_dz.reshape((dL_dz.shape[0],1))

    dz_dw = X
    dL_dw = dL_dz * dz_dw
    dL_dw_avg = np.mean(dL_dw, axis=0)

    dz_db = 1
    dL_db = dL_dz * dz_db
    dL_db_avg = np.mean(dL_db, axis=0)

    return dL_dw_avg, dL_db_avg

############## Main ##############

# Construct dataset
n_train = 10000
n_test = 2000
n_features = 10
coeffs_true = (np.arange(n_features) - (n_features/2-0.5)) / 2
X_train, y_train = generate_dataset(coeffs_true, n_train)
X_test, y_test = generate_dataset(coeffs_true, n_test)

# Fit model
# The model is a simple logistic regression:
#   output = sigmoid( w*x_i + b ),
# where w is a weight vector of shape (n_features,), and b is a scalar bias
# and x_i is the ith sample, of shape (n_features,)
n_epochs = 1000
learning_rate = 0.1
w = np.random.rand(n_features)
b = np.random.rand(1)
results = defaultdict(list)
print(f'Running gradient descent for {n_epochs} epochs...')
for n_epoch in tqdm.tqdm(range(n_epochs)):

    # Forward pass
    y_pred = forward_pass(X_train, w, b)

    # Compute loss over all samples
    results['loss'].append(loss(y_pred, y_train))

    # Backpropagation: compute gradients of loss wrt w,b -- average over all samples
    dL_dw, dL_db = backpropagate(X_train, w, b, y_train)

    # Update w,b
    w -= learning_rate * dL_dw
    b -= learning_rate * dL_db

# Test model
y_pred = forward_pass(X_test, w, b)
w = w * np.mean(coeffs_true / w) # Normalize coeffs 

# Print coeffs -- normalize
print()
print(f'Results on test set:')
print(f'  coeffs_true = {coeffs_true}')
print(f'  coeffs_pred = {w}')
print(f'  b_true = 0')
print(f'  b_pred = {b}')

# Make some plots
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Plot ROC curve
thresholds = np.linspace(0,1,100) # Output of sigmoid is in [0,1]
for threshold in thresholds:
    y_pred_thresholded = y_pred > threshold

    true_positives = np.sum(y_pred_thresholded*y_test)
    total_positives = np.sum(y_test)
    results['tpr'].append(true_positives/total_positives)

    false_positives = np.sum(y_pred_thresholded*np.abs(1-y_test))
    total_negatives = np.sum(np.abs(1-y_test))
    results['fpr'].append(false_positives/total_negatives)

plt.plot([0,1],[0,1], 'k--')
plt.plot(results['fpr'], results['tpr'])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.savefig(os.path.join(output_dir, 'roc.png'))
plt.close()

# Plot loss
plt.plot(results['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(output_dir, 'loss.png'))
plt.close()