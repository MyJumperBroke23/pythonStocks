import numpy as np
import torch
import pandas as pd

seed = 1  # Sample seed
df = pd.read_csv("data/AAPL60.csv")
df["indicies"] = df.index
train = df.sample(frac=0.9, replace=True, random_state=seed)
print(train[0]['indicies'])
exit()
test = df.loc[~df.index.isin(train.index), :]
s = df['close']
l = len(s)
j = 100  # Use past j days as predictor
xtemp = []
ytemp = []
for i in range(0, l - j):
    xtemp.append(np.array(s[i:i + j].values))
    ytemp.append(s.get(i + j))

trainx = np.array(xtemp, dtype=np.float32)
trainy = np.array(ytemp, dtype=np.float32)
trainy = trainy.reshape(-1, 1)
trainx = trainx / 300
trainy = trainy / 300


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
    inputs = torch.from_numpy(trainx).requires_grad_()
    labels = torch.from_numpy(trainy)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = critereon(outputs, labels)
    print(loss)
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))
# print(model(Variable(torch.from_numpy(xdata))))
