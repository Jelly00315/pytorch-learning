# Goal：use linear model： y_hat = w * x + b
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# data set
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([3, 5, 7, 9, 11])

# define a model
def forward(x):
    return w * x + b

# define a loss function
def loss(x, y):
    y_pred = forward(x) # y_hat
    return (y_pred - y) ** 2 # mse

# empty lists
w_list = []
b_list = []
mse_list = []

# calculate the loss 
for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(0.0, 4.1, 0.1):
        print ("w=", w, "b=", b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print('MSE=', l_sum/5)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum/5)

# draw the graph

# 3D plot, from m, b, mse
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(w_list, b_list, mse_list, c=mse_list, cmap='viridis')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
plt.show()

# present by colour, contour plot, from m, b, mse
plt.figure()
plt.contourf(np.array(w_list).reshape(41, 41), np.array(b_list).reshape(41, 41), np.array(mse_list).reshape(41, 41), levels=50, cmap='viridis')
plt.xlabel('w')
plt.ylabel('b')
plt.colorbar(label='MSE')
plt.show()