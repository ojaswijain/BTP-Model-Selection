import numpy as np
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Function to generate a sparse signal
def generate_sparse_signal(signal_size, sparsity):
    sparse_signal = np.zeros(signal_size)
    sparse_indices = np.random.choice(signal_size, sparsity, replace=False)
    sparse_signal[sparse_indices] = np.random.randn(sparsity)
    return sparse_signal

# Function to generate a random Gaussian matrix
def generate_gaussian_matrix(num_measurements, signal_size):
    return np.random.randn(num_measurements, signal_size) / np.sqrt(num_measurements)

# Function to take measurements
def take_measurements(measurement_matrix, sparse_signal):
    return np.dot(measurement_matrix, sparse_signal)

# Function to reconstruct the sparse signal using compressive sensing
def reconstruct_signal(measurement_matrix, measurements, lambda_val):
    signal_size = measurement_matrix.shape[1]
    initial_guess = np.zeros(signal_size)

    # Define the objective function for reconstruction
    def objective_function(x):
        return np.linalg.norm(measurements - np.dot(measurement_matrix, x))**2 + lambda_val * np.linalg.norm(x, ord=1)

    # Define the constraint function (Ax = b)
    def constraint_function(x):
        return np.dot(measurement_matrix, x) - measurements

    # Solve the optimization problem
    result = minimize(objective_function, initial_guess, constraints={'type': 'eq', 'fun': constraint_function})

    # Reconstructed signal
    reconstructed_signal = result.x
    return reconstructed_signal

# Parameters
signal_size = 128
num_signals = 100
sparsity = int(0.05 * signal_size)
lambda_range = [0.001, 0.01, 0.1, 1, 10, 100]

# Generate a set of sparse signals
X = np.column_stack([generate_sparse_signal(signal_size, sparsity) for _ in range(num_signals)])

# Generate a random Gaussian matrix
num_measurements = 16
phi = generate_gaussian_matrix(num_measurements, signal_size)

# Take measurements for each sparse signal
Y = np.dot(phi, X)

# Split data into training and validation sets
X_train, X_valid, Y_train, Y_valid = train_test_split(X.T, Y.T, test_size=0.15, random_state=42)

# Transpose back to the original shape
X_train, X_valid, Y_train, Y_valid = X_train.T, X_valid.T, Y_train.T, Y_valid.T

# Search for the best lambda
best_lambda = None
best_validation_error = float('inf')

for lambda_val in lambda_range:
    # Reconstruct signals on the training set using the current lambda
    reconstructed_signals_train = np.column_stack([reconstruct_signal(phi, Y_train[:, i], lambda_val) for i in range(X_train.shape[1])])
    
    # Evaluate on the validation set
    reconstructed_signals_valid = np.column_stack([reconstruct_signal(phi, Y_valid[:, i], lambda_val) for i in range(X_valid.shape[1])])
    validation_error = np.linalg.norm(reconstructed_signals_valid - X_valid)

    # Update best_lambda if the current lambda leads to better performance
    if validation_error < best_validation_error:
        best_validation_error = validation_error
        best_lambda = lambda_val

# Train the final model using the best lambda on the entire training set
final_reconstructed_signals = np.column_stack([reconstruct_signal(phi, Y[:, i], best_lambda) for i in range(X.shape[1])])

print('Best lambda: {}'.format(best_lambda))

import os

# ... (previous code remains unchanged)

# Create a folder to save plots
output_folder = "reconstruction_plots"
os.makedirs(output_folder, exist_ok=True)

# Visualize the results for each signal and save plots
for i in range(X.shape[1]):
    # Reconstruct the signal for the i-th column of Y
    reconstructed_signal_i = reconstruct_signal(phi, Y[:, i], best_lambda)
    
    # Plot original signal and reconstructed signal
    plt.figure()
    plt.plot(X[:, i], label='Original Signal')
    plt.plot(reconstructed_signal_i, label='Reconstructed Signal')
    plt.title(f'Signal {i+1}')
    plt.legend()
    
    # Save the plot in the output folder
    plt.savefig(os.path.join(output_folder, f'reconstructed_signal_{i+1}.png'))
    plt.close()

# Optionally, display a message indicating that the plots are saved
print(f"Reconstructed signal plots saved in the '{output_folder}' folder.")

# # Visualize the results
# plt.plot(X[:, 0], label='Original Signal')
# plt.plot(final_reconstructed_signals[:, 0], label='Reconstructed Signal')
# plt.legend()
# plt.show()