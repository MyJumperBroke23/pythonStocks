from builtins import enumerate
import numpy as np
import torch
from torch.autograd import Variable
import pandas as pd

df = pd.read_csv("data/AAPL60.csv")
s = df['close']
l = len(s)
j = 100  # Use past j days as predictor
xtemp = []
ytemp = []
for i in range(0, l - j):
    xtemp.append(np.array(s[i:i + j].values))
    ytemp.append(s.get(i + j))

xdata = np.array(xtemp, dtype=np.float32)
ydata = np.array(ytemp, dtype=np.float32)
ydata = ydata.reshape(-1,1)
xdata=xdata/300
ydata=ydata/300
# x_values = [i for i in range(11)]
# x_train = np.array(x_values, dtype=np.float32)
# x_train = x_train.reshape(-1, 1)
# y_values = [2 * i + 1 for i in x_values]
# y_train = np.array(y_values, dtype=np.float32)
# y_train = y_train.reshape(-1, 1)


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegressionModel(j, 1)
critereon = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(500):
    inputs = torch.from_numpy(xdata).requires_grad_()
    labels = torch.from_numpy(ydata)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = critereon(outputs, labels)
    print(loss)
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))
# print(model(Variable(torch.from_numpy(xdata))))
