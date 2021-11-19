import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt


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
    cost = -np.sum(np.log(1 - a[ind, :]))
    # add cost of label 1 points
    ind = np.argwhere(y == 1)[:, 0]
    cost -= np.sum(np.log(a[ind, :]))
    # compute cross-entropy
    return cost / y.size


# gradient descent function
def gradient_descent(g, step, max_its, w, p):
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


# logistric regression script
# create data set (random)
# each class separately
# x for income
# y for elections candidate
P_per_class = 20
x0 = np.random.rand(P_per_class)
y0 = np.zeros(P_per_class)
x1 = 2 + np.random.rand(P_per_class)
y1 = np.ones(P_per_class)
# append classes
x = np.append(x0, x1)
y = np.append(y0, y1)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
# plot dataset
plt.scatter(x, y)
plt.xlabel('income')
plt.ylabel('candidate')


def c(t):
    c = cross_entropy(t, x, y)
    return c


iter = 100
w = np.array([[1.], [1.]])
a, b = gradient_descent(c, 1, iter, w, 0)
plt.figure(0)
plt.plot(b)
plt.figure(1)
xp = np.array([np.linspace(0, 3, 20)])
xp = xp.reshape(-1, 1)
plt.plot(xp, sigmoid(model(xp, a[iter])))
plt.show()
