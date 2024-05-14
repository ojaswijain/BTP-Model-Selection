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

# Function to generate a random measurement matrix
def generate_measurement_matrix(num_measurements, signal_size):
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
signal_size = 128 * 128
sparsity = int(0.05 * signal_size)
num_measurements = int(0.85 * signal_size)
lambda_val = 0.1

# Generate a sparse signal
sparse_signal = generate_sparse_signal(signal_size, sparsity)

# Generate a random measurement matrix
measurement_matrix = generate_measurement_matrix(num_measurements, signal_size)

# Take measurements
measurements = take_measurements(measurement_matrix, sparse_signal)

# Split the data into train and validation sets
measurement_matrix_train, measurement_matrix_valid, measurements_train, measurements_valid = train_test_split(
    measurement_matrix, measurements, test_size=0.15, random_state=42
)

# Reconstruct the signal using compressive sensing with training data
reconstructed_signal_train = reconstruct_signal(measurement_matrix_train, measurements_train, lambda_val)

# Evaluate the performance on the validation set
reconstructed_signal_valid = np.dot(measurement_matrix_valid, reconstructed_signal_train)
validation_error = np.linalg.norm(reconstructed_signal_valid - sparse_signal)

# Visualize the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(sparse_signal.reshape(128, 128), cmap='viridis')
plt.title('Original Sparse Signal')

plt.subplot(1, 3, 2)
plt.imshow(reconstructed_signal_train.reshape(128, 128), cmap='viridis')
plt.title('Reconstructed Signal (Training)')

plt.subplot(1, 3, 3)
plt.imshow(reconstructed_signal_valid.reshape(128, 128), cmap='viridis')
plt.title('Reconstructed Signal (Validation)')

plt.show()

print(f"Validation Error: {validation_error}")
