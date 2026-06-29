# prepare dataset

import numpy as np
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32) # load data from csv file, gpu widely support float 32 but no 16
x_data = torch.from_numpy(xy[:,:-1]) # all rows, all columns except last column(-1)
y_data = torch.from_numpy(xy[:, [-1]]) # all rows, last column

# define model
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6) # 8D -> 6D
        self.linear2 = torch.nn.Linear(6, 4) # 6D -> 4D
        self.linear3 = torch.nn.Linear(4, 1) # 4D -> 1D
        self.sigmoid = torch.nn.Sigmoid() # here is your activation func, you can also change to ReLU or else

    def forward(self, x):
        x = self.sigmoid(self.linear1(x)) #O1
        x = self.sigmoid(self.linear2(x)) #O2
        x = self.sigmoid(self.linear3(x)) #y_hat, all use x to avoid dimension problems
        return x

model = Model()

# construct loss and optimizer

criterion = torch.nn.BCELoss(size_average = False) # binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# training loop
# although we use mini-Batch, but not in here but by Dataloader
# here we just put all data in it
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# there are multiple activation functions, use as you need