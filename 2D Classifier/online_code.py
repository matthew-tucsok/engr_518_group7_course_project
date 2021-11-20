import random
import numpy as np
import matplotlib.pyplot as plt


def linear_regression(inputs, targets, epochs, learning_rate):
    """
    A utility function to run linear regression and get weights and bias
    """
    costs = []  # A list to store losses at each epoch
    values_count = inputs.shape[1]  # Number of values within a single input
    size = inputs.shape[0]  # Total number of inputs
    weights = np.zeros((values_count, 1))  # Weights
    bias = 0  # Bias

    for epoch in range(epochs):
        predicted = np.dot(inputs, weights) + bias  # Calculating the predicted values
        loss = predicted - targets  # Calculating the individual loss for all the inputs
        d_weights = np.dot(inputs.T, loss) / (2 * size)  # Calculating gradient
        d_bias = np.sum(loss) / (2 * size)  # Calculating gradient
        weights = weights - (learning_rate * d_weights)  # Updating the weights
        bias = bias - (learning_rate * d_bias)  # Updating the bias
        cost = np.sqrt(np.sum(loss ** 2) / (2 * size))  # Root Mean Squared Error Loss or RMSE Loss
        costs.append(cost)  # Storing the cost
        print(f"Iteration: {epoch + 1} | Cost/Loss: {cost} | Weight: {weights} | Bias: {bias}")

    return weights, bias, costs


def plot_test(inputs, targets, weights, bias):
    """
    A utility function to test the weights
    """
    predicted = np.dot(inputs, weights) + bias
    predicted = predicted.astype(int)
    plt.plot(predicted, [i for i in range(len(predicted))], color=np.random.random(3), label="Predictions",
             linestyle="None", marker="x")
    plt.plot(targets, [i for i in range(len(targets))], color=np.random.random(3), label="Targets", linestyle="None",
             marker="o")
    plt.xlabel("Indexes")
    plt.ylabel("Values")
    plt.title("Predictions VS Targets")
    plt.legend()
    plt.show()


def rmse(inputs, targets, weights, bias):
    """
    A utility function to calculate RMSE or Root Mean Squared Error
    """
    predicted = np.dot(inputs, weights) + bias
    mse = np.sum((predicted - targets) ** 2) / (2 * inputs.shape[0])
    return np.sqrt(mse)


def generate_data(m, n, a, b):
    """
    A function to generate training data, training labels, testing data, and testing inputs
    """
    x, y, tx, ty = [], [], [], []

    for i in range(1, m + 1):
        x.append([float(i)])
        y.append([float(i) * a + b])

    for i in range(n):
        tx.append([float(random.randint(1000, 100000))])
        ty.append([tx[-1][0] * a + b])

    return np.array(x), np.array(y), np.array(tx), np.array(ty)


learning_rate = 0.0001  # Learning rate
epochs = 200000  # Number of epochs
a = 0.5  # y = ax + b
b = 2.0  # y = ax + b
inputs, targets, train_inputs, train_targets = generate_data(300, 50, a, b)
weights, bias, costs = linear_regression(inputs, targets, epochs, learning_rate)  # Linear Regression
indexes = [i for i in range(1, epochs + 1)]
plot_test(train_inputs, train_targets, weights, bias)  # Testing
print(f"Weights: {[x[0] for x in weights]}")
print(f"Bias: {bias}")
print(f"RMSE on training data: {rmse(inputs, targets, weights, bias)}")  # RMSE on training data
print(f"RMSE on testing data: {rmse(train_inputs, train_targets, weights, bias)}")  # RMSE on testing data
plt.plot(indexes, costs)
plt.xlabel("Epochs")
plt.ylabel("Overall Cost/Loss")
plt.title(f"Calculated loss over {epochs} epochs")
plt.show()
