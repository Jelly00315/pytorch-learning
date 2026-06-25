import numpy as np # numpy is a library for scientific computing in Python. It provides support for arrays, matrices, and many mathematical functions to operate on these data structures.
import matplotlib.pyplot as plt

x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 4, 6, 8, 10])

# make initial guess for w
w = 1.0

# define a linear model
def forward(x):
    return w * x
    
# define a cost function
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)
# if you are doing Stochastic gradient descent, you only need to use loss
# def loss(x, y):
#     y_pred = forward(x)
#     return (y_pred - y) ** 2
# and in this case, you can simply do
# def gradient(x, y):
#    return 2 * x * (forward(x) - y)
# also:
# for epoch in range(100):
#   for x_val, y_val in zip(x_data, y_data):
#       grad = gradient(x, y)
#       w = w - 0.01 * grad
#       print('\tgrad:', x, y, grad)
#       l = loss(x, y)
#   print("progress:", epoch, "w=", w, "loss=", 1)

# define a gradient function
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        grad += 2 * x * (y_pred - y)
    return grad / len(xs)

print('Predict (before training):', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)
print('Predict (after training):', 4, forward(4))