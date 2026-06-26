# why backward propagation work? each step get one gradient for one linear movement, 
# and because of chain rule, combination of linear movement => multiplication of these gradients. 
# we do this for flexibility, when we modify one parameter, we can still use the same model
# sort of like Lego, we use blocks 


import torch

x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [2.0, 4.0, 6.0, 8.0, 10.0]

w = torch.tensor([1.0])
w.requires_grad = True # True: need to calculate the gradient of w

def forward(x): # forward(x) = y_hat
    return w * x # we are building the computation model, thus we can directly use tensor(instead of the data)

def loss(x, y): # calculate loss from y_hat - y
    y_pred = forward(x)
    return (y_pred - y) ** 2

    print('Predict (before training)', 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y) # calculate loss
        l.backward() # backpropagation, calculate the gradient of w
        print('\tgrad:', x, y, w.grad.item()) # w.grad is a tensor, is a calculation model, so you have to add .data
        w.data = w.data - 0.01 * w.grad.data # update w
        w.grad.data.zero_() # the grad computed by backward() is accumulated, so you have to clear it after each step
    print("progress:", epoch, "w=", w.data.item(), "loss=", l.item())
print('Predict (after training)', 4, forward(4).item())
 