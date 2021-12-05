import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit


@jit(nopython=True, parallel=True)
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


@jit(nopython=True, parallel=True)
def regularized_cross_entropy_cost(y, weights, preds, lam, log_eps=1e-9):
    return -1 / (len(y)) * np.sum(y * np.log(preds + log_eps) + (1 - y) * np.log(
        1 - preds + log_eps))


def find_gradient(weights, x_bar, y, eps, lam):
    gradients = []
    preds = sigmoid(np.dot(x_bar, weights))
    fx = regularized_cross_entropy_cost(y, weights, preds, lam)
    temp_weights = weights.copy()
    for weight in range(len(weights)):
        temp_weights[weight] += eps
        preds = sigmoid(np.dot(x_bar, temp_weights))
        fx_plus_eps = regularized_cross_entropy_cost(y, temp_weights, preds, lam)
        gradients.append((fx_plus_eps - fx) / eps)
        temp_weights[weight] -= eps
    return np.array(gradients)


def find_gradient_analytically(weights, x_bar, y, eps, lam):
    return np.array(-1 / y.shape[0]) * np.matmul(x_bar.T, y - sigmoid(np.matmul(x_bar, weights)))


@jit(nopython=True, parallel=True)
def fast_find_gradient(weights, x_bar, y, eps, lam):
    log_eps = 1e-9
    gradients = []
    preds = 1 / (1 + np.exp(-(np.dot(x_bar, weights))))
    fx = regularized_cross_entropy_cost(y, weights, preds, lam)
    temp_weights = weights.copy()
    for weight in range(weights.shape[0]):
        temp_weights[weight] += eps
        preds = 1 / (1 + np.exp(-(np.dot(x_bar, temp_weights))))
        fx_plus_eps = -1 / (y.shape[0]) * np.sum(
            y * np.log(preds + log_eps) + (1 - y) * np.log(1 - preds + log_eps) + lam * np.linalg.norm(weights[1:]))
        gradients.append((fx_plus_eps - fx) / eps)
        temp_weights[weight] -= eps
    return np.array(gradients)


class Model:
    def __init__(self, samples, learning_rate, epsilon, batch_size=32):
        self.x, self.y = zip(*samples)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        num_samples = len(self.y)
        self.x_bar = np.hstack((np.ones([num_samples, 1]), self.x))
        self.weights = np.zeros([self.x_bar.shape[1]])

        self.lam = 0
        self.eps = epsilon
        self.alpha = learning_rate
        self.iterations = 1000
        self.batch_size = batch_size

        self.num_batches = int(num_samples / self.batch_size)
        total_rows = self.num_batches * self.batch_size
        temp_x_bar = self.x_bar[:total_rows, :].copy()
        temp_y = self.y[:total_rows].copy()
        self.x_bar_batches = np.split(temp_x_bar, self.num_batches, axis=0)
        self.y_batches = np.split(temp_y, self.num_batches, axis=0)
        self.gradient_descent()

    def gradient_descent(self):
        start = time.time()
        total = 0
        history = []
        for i in range(self.iterations):
            if i % 1000 == 0:
                print('Epoch:', i)
            preds = sigmoid(np.dot(self.x_bar, self.weights))
            cost = regularized_cross_entropy_cost(self.y, self.weights, preds, self.lam)
            if i % 1000 == 0:
                print('Cost:', cost)
            end = time.time()
            duration = end - start
            if i % 1000 == 0:
                print('Execution time:', duration)
            total += duration
            start = time.time()
            for batch in range(self.num_batches):
                gradient = find_gradient_analytically(self.weights, self.x_bar_batches[batch], self.y_batches[batch], self.eps,
                                                      self.lam)
                gradient_norm = np.linalg.norm(gradient)
                self.weights -= self.alpha / (i + 1) * gradient / gradient_norm
            history.append((cost, self.weights, i))
        print('Total execution for', self.iterations, 'Epochs:', total)
        costs, weights, indices = zip(*history)
        plt.plot(range(self.iterations), costs)
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.title('Cost vs Iterations')
        plt.show()
        history.sort(key=lambda tup: tup[0])
        print('Best weights after epoch', history[0][2], 'with a cost of', history[0][0])
        self.weights = history[0][1]

    def accuracy(self, test_samples):
        x_tests, y_tests = zip(*test_samples)
        x_bar_tests = np.hstack((np.ones([len(y_tests), 1]), x_tests))
        predictions = sigmoid(np.dot(x_bar_tests, self.weights))
        return Model.confusion_matrix(y_tests, predictions)

    @staticmethod
    def confusion_matrix(actuals, predictions):
        thresh = predictions.round()
        c11 = np.sum(np.logical_not(np.logical_or(actuals, thresh)))
        c12 = np.sum(np.logical_and(np.logical_not(actuals), thresh))
        c21 = np.sum(np.logical_and(actuals, np.logical_not(thresh)))
        c22 = np.sum(np.logical_and(actuals, thresh))
        matrix = np.array([[c11, c12], [c21, c22]])
        return matrix
