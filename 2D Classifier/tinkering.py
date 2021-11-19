import numpy as np
from autograd import grad


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = b - 1
print(c)

x_bar = np.array([[1, 1], [2, 6], [3, 7]])
weights = np.array([1, 2, 3])
y = np.array([1, 0])


def cross_entropy(w):
    return -1 / len(y) * np.sum(y * np.log(sigmoid(np.dot(w, x_bar))) + (1 - y) * np.log(1 - sigmoid(np.dot(w, x_bar))))


t1 = np.log(sigmoid(14))
t2 = np.log(1 - sigmoid(34))
t3 = t1 + t2
t4 = -t3 / 2
print(t4)

w = np.array([1, 1, 2, 3])
w_new = w[1:]
print(w_new)

# -1/len(self.y) * np.sum(self.y * np.log(Model.sigmoid(np.dot(weights, self.x_bar))) + (1-self.y)*np.log(1-Model.sigmoid(np.dot(weights, self.x_bar)))) + self.lam*np.linalg.norm(weights[1:])
