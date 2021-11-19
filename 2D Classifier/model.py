import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def model(x_p, w):
    # compute linear comb and return
    a = w[0] + np.dot(x_p, w[1:])
    return a


# define sigmoid
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


# cross-entropy function
def cross_entropy(w, x, y):
    # compute sigmoid of model
    a = sigmoid(model(x, w))
    # compute cost of label 0 points
    ind = np.argwhere(y == 0)[:, 0]
    cost = -np.sum(np.log(1 - a[ind] + 1e-9))
    # add cost of label 1 points
    ind = np.argwhere(y == 1)[:, 0]
    cost -= np.sum(np.log(a[ind]) + 1e-9)
    # compute cross-entropy
    return cost / y.size


class Model:
    def __init__(self, samples):
        self.x, self.y = zip(*samples)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.x_bar = np.hstack((np.ones([len(self.y), 1]), self.x))
        self.weights = np.zeros([self.x_bar.shape[1]])
        self.lam = 0
        self.gradient_descent()

    @staticmethod
    def sigmoid(t):
        return 1 / (1 + np.exp(-t))

    def c(self, t):
        c = self.regularized_cross_entropy_cost()
        return c

    def regularized_cross_entropy_cost(self):
        cost = -1 / float(len(self.y)) * np.sum(
            self.y * np.log(Model.sigmoid(np.dot(self.x_bar, self.weights)) + 1e-9) + (1 - self.y) * np.log(
                1 - Model.sigmoid(np.dot(self.x_bar, self.weights)) + 1e-9)) + self.lam * np.linalg.norm(
            self.weights[1:])
        return cost

    def gradient_descent(self):
        grad_cross_entropy = grad(self.c)
        yeezy = grad_cross_entropy(self.weights)
        x = 1


class InClassModel:
    def __init__(self, samples):
        self.x, self.y = zip(*samples)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.x_bar = np.hstack((np.ones([len(self.y), 1]), self.x))
        self.weights = np.ones([self.x_bar.shape[1]])
        self.lam = 0
        self.iterations = 1000

        def c(t):
            c = cross_entropy(t, self.x, self.y)
            return c

        weight_history, cost_history = self.gradient_descent(c, 'd', self.iterations, self.weights)
        self.final_weights = weight_history[self.iterations]
        plt.figure(0)
        plt.plot(cost_history)
        plt.show()

    def gradient_descent(self, g, step, max_its, w):
        # compute gradient
        gradient = grad(g)
        # gradient descent loop
        weight_history = [w]  # weight history container
        cost_history = [g(w)]  # cost history container
        for k in range(max_its):
            # eval gradient
            grad_eval = gradient(w)
            grad_eval_norm = grad_eval / np.linalg.norm(grad_eval)
            # take grad descent step
            if step == 'd':  # diminishing step
                alpha = 1 / (k + 1)
            else:  # constant step
                alpha = step
            w = w - alpha * grad_eval_norm
            # record weight and cost
            weight_history.append(w)
            cost_history.append(g(w))
        return weight_history, cost_history

    def accuracy(self, test_samples):
        x_tests, y_tests = zip(*test_samples)
        predictions = sigmoid(model(x_tests, self.final_weights))
        return confusion_matrix(y_tests, predictions)
