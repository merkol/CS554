## scikit-learn mlp with 8 hidden layers
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


train_data = np.loadtxt("data_2024/train.csv", delimiter=",", skiprows=1)
test_data = np.loadtxt("data_2024/test.csv", delimiter=",", skiprows=1)

# sort the data
train_data = train_data[train_data[:, 0].argsort()]
test_data = test_data[test_data[:, 0].argsort()]

# split the data into x and y
x_train, y_train = train_data[:, 0].reshape(-1,1), train_data[:, 1].reshape(-1,1)
x_test, y_test = test_data[:, 0].reshape(-1,1), test_data[:, 1].reshape(-1,1)

# mlp = MLPRegressor(hidden_layer_sizes=(8,), max_iter=11000, learning_rate_init=0.01, activation='logistic', verbose=True, batch_size=100, random_state=42)
# tick = time.time()
# mlp.fit(x_train, y_train.ravel())
# tock = time.time()

# print(f"Training time: {tock - tick}")

# y_pred = mlp.predict(x_test)
# print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
# lin = np.linspace(x_train.min(), x_train.max(), 1000).reshape(-1, 1)
# y_pred = mlp.predict(lin)


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from sklearn.metrics import mean_squared_error
import time


# Define Xavier initialization function
def xavier_init(n_inputs, n_outputs):
    variance = 1.0 / (n_inputs + n_outputs)
    std_dev = np.sqrt(variance)
    return np.random.randn(n_inputs, n_outputs) * std_dev

# Define custom implementation of the neural network
class Net:
    def __init__(self):
        self.fc1_weight = xavier_init(1, 8)
        self.fc1_bias = np.zeros(8)
        self.fc2_weight = xavier_init(8, 1)
        self.fc2_bias = np.zeros(1)

    def forward(self, x):
        x = sigmoid(np.dot(x, self.fc1_weight) + self.fc1_bias)
        x = np.dot(x, self.fc2_weight) + self.fc2_bias
        return x

# Define Mean Squared Error loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


# Create an instance of the neural network
net = Net()

# Define training parameters
lr = 0.01
num_epochs = 10000
losses = []

# Train the neural network
tick = time.time()
for epoch in range(num_epochs):
    output = net.forward(x_train)
    loss = mean_squared_error(y_train, output)
    losses.append(loss)
    
    # Backpropagation (no gradients explicitly computed here)
    grad_output = 2 * (output - y_train) / len(x_train)
    grad_fc2_weight = np.dot(sigmoid(np.dot(x_train, net.fc1_weight) + net.fc1_bias).T, grad_output)
    grad_fc2_bias = np.sum(grad_output, axis=0)
    grad_fc1 = np.dot(grad_output, net.fc2_weight.T) * sigmoid(np.dot(x_train, net.fc1_weight) + net.fc1_bias) * (1 - sigmoid(np.dot(x_train, net.fc1_weight) + net.fc1_bias))
    grad_fc1_weight = np.dot(x_train.T, grad_fc1)
    grad_fc1_bias = np.sum(grad_fc1, axis=0)
    
    # Update weights and biases
    net.fc2_weight -= lr * grad_fc2_weight
    net.fc2_bias -= lr * grad_fc2_bias
    net.fc1_weight -= lr * grad_fc1_weight
    net.fc1_bias -= lr * grad_fc1_bias

tock = time.time()

print(f"Training time: {tock - tick}")

# Generate predictions on test data
x_test = np.linspace(x_train.min(), x_train.max(), 1000).reshape(-1, 1)
y_pred = net.forward(x_test)


# Plot training data and predictions
plt.scatter(x_train, y_train, color='black', label='Training Data')
plt.plot(x_test, y_pred, color='red', label='MLP Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Plot loss over iterations
plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

# # plot test data and predictions as a function of x
# plt.scatter(x_test, y_test, color='black', label='Test Data')
# plt.plot(lin, y_pred, color='red', label='MLP Prediction')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()



# # plot training data and predictions as a function of x
# plt.scatter(x_train, y_train, color='black', label='Training Data')
# plt.plot(lin, y_pred, color='red', label='MLP Prediction')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()

# ## new figure for loss curve
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(mlp.loss_curve_)
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Loss')
# plt.show()



