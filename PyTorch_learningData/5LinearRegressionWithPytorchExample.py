# prepare dataset

import torch
x_data = torch.tensor([[1.0], [2.0], [3.0]]) # this data should be a matrix(for computing)
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# design model using class
from pyexpat import model

class LinearModel(torch.nn.Module): # our model class inherits from torch.nn.Module(which is base class for all neural network modules)
    def __init__(self): # intial your object, and define the layers you want in your model
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1) # size of input sample = 1, size of output sample = 1, bias = true by default
        # class nn.linear contains two member Tensors: weight and bias

    def forward(self, x): # module would automatically build the backward function
        y_pred = self.linear(x) # callable object. you can also use __call__(self, *args, **kwargs) to call the forward function
        return y_pred
    
model = LinearModel()


# construct loss and optimizer

criterion = torch.nn.MSELoss(size_average = False) # size_average = False means the loss will be summed over all observations, not averaged.   
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) # lr = learning rate. class torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False) 

# training loop

for epoch in range(100): 
    # forward
    y_pred = model(x_data) # y_hat
    loss = criterion(y_pred, y_data) # loss
    print(epoch, loss)
    # backward
    optimizer.zero_grad() 
    loss.backward() # backpropagation, compute gradients
    # update
    optimizer.step() 




# output weight and bias
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

# test model
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)


# there are lots of optimizer in linear regression
# torch.optim.Adagrad
# torch.optim.Adam
# torch.optim.Adamax
# torch.optim.ASGD
# torch.optim.LBFGS
# torch.optim.RMSprop
# torch.optim.Rprop
# torch.optim.SGD