import numpy as np
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

np.random.seed(1)

output_folder = "reconstruction_plots"
os.makedirs(output_folder, exist_ok=True)

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
def take_measurements(measurement_matrix, sparse_signal, mean, var):
    return np.dot(measurement_matrix, sparse_signal) + np.random.normal(mean, var, measurement_matrix.shape[0])

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
# num_measurements = 100
# sparsity = int(0.1 * signal_size)
set_of_measurements = [50, 75, 100]
sparsities = [int(0.05 * signal_size), int(0.1 * signal_size), int(0.2 * signal_size), int(0.5 * signal_size)]
lambda_range = [0.001, 0.01, 0.1, 1, 10, 100]

#save validation errors for each sparsity and number of measurements
validation_errors = np.zeros((len(set_of_measurements), len(sparsities)))

for num_measurements in set_of_measurements:
    for sparsity in sparsities:

        # Generate a sparse signal
        x = generate_sparse_signal(signal_size, sparsity)

        # Generate a random Gaussian matrix
        phi = generate_gaussian_matrix(num_measurements, signal_size)

        #define mean and variance
        mean = 0
        k = 0.05
        var = (1/num_measurements)*sum(abs(np.dot(phi[i,:], x)) for i in range(num_measurements))*k
        # Take measurements
        y = take_measurements(phi, x, mean, var)

        # Split data into training and validation sets
        phi_train, phi_valid, y_train, y_valid = train_test_split(phi, y, test_size=0.15, random_state=42)

        # Search for the best lambda
        best_lambda = None
        best_validation_error = float('inf')

        for lambda_val in lambda_range:
            # Reconstruct the signal for each measurement
            x_train = reconstruct_signal(phi_train, y_train, lambda_val)

            # Compute the validation error
            validation_error = np.linalg.norm(y_valid - np.dot(phi_valid, x_train))**2

            # Update best_lambda if the current lambda leads to better performance
            if validation_error < best_validation_error:
                best_validation_error = validation_error
                best_lambda = lambda_val

        # Display the best lambda
        # print('Best lambda: {}'.format(best_lambda))
        x_reconstructed = reconstruct_signal(phi, y, best_lambda)
        validation_errors[set_of_measurements.index(num_measurements), sparsities.index(sparsity)] = np.linalg.norm(x_reconstructed - x)/np.linalg.norm(x)

        # Visualize the results, making subplots for the original and reconstructed signals
        plt.subplot(2, 1, 1)
        plt.plot(x, color='b')
        plt.title('Original signal')
        plt.subplot(2, 1, 2)
        plt.plot(x_reconstructed, color='r')
        plt.title('Reconstructed signal')
        plt.tight_layout()  
        plt.savefig(os.path.join(output_folder, 'S_{}_m_{}.png'.format(sparsity, num_measurements)))
        plt.clf()

set_of_measurements = np.array(set_of_measurements)
sparsities = np.array(sparsities)

print(validation_errors)
# Plot the validation errors against the number of measurements and sparsity as a heatmap
plt.imshow(validation_errors, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.xticks(np.arange(len(sparsities)), sparsities)
plt.yticks(np.arange(len(set_of_measurements)), set_of_measurements)
plt.xlabel('Sparsity')  
plt.ylabel('Number of Measurements')
plt.title('Validation Error')
plt.savefig(os.path.join(output_folder, 'validation_error.png'))
plt.show()

