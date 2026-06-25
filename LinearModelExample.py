import numpy as np # numpy is a library for scientific computing in Python. It provides support for arrays, matrices, and many mathematical functions to operate on these data structures.
import matplotlib.pyplot as plt

x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 4, 6, 8, 10])

# define a model
def forward(x):
    return w * x 

# define a loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# create empty lists to store the values of w, b, and loss
w_list = []
mse_list = []

# calculate the loss for different values of w
for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data): # zip the 2 lists together, so that we can iterate through them in parallel
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)

# to draw the graph, we use: 
# plt.plot(w_list, mse_list)
# plt.ylabel('Loss')
# plt.xlabel('w')
# plt.show()
