# classification: you may use the MINIST dataset(handwritten digtis) and CIFAR-10 dataset(images of 10 classes of objects)
import torch
from pyexpat import model
import torch.nn.functional as F

# prepare dataset
x_data = torch.tensor([[1.0], [2.0], [3.0]]) # this data should be a matrix(for computing)
y_data = torch.tensor([[0.0], [0.0], [1.0]]) # 0: class 0, 1: class 1

# design model using class
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1) # size of input sample = 1, size of output sample = 1, bias = true by default

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x)) # apply sigmoid function to the linear output, non-linear transformation
        return y_pred
    
model = LogisticRegressionModel()

# construct loss and optimizer
criterion = torch.nn.BCELoss(size_average = False) # binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# training loop
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# draw graph
import numpy as np
import matplotlib.pyplot as plt

# then write according to dimension and features of dataset