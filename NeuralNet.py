import torch
import torch.nn as nn
import numpy as np
from dataio.py import readCSV

data = readCSV("data/AAPL60.csv")
np.flip(A, 0)

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=1,
                 num_layers=2, drop_p = 0.3, linear_dim = 16):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_p = drop_p
        self.output_dim = output_dim
        self.linear_dim = linear_dim

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear1 = nn.Linear(self.hidden_dim, self.linear_dim)
        self.linear2 = nn.Linear(self.linear_dim, self.output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, 1, self.hidden_dim),
                torch.zeros(self.num_layers, 1, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input, None)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear1(lstm_out[:, -1, :])
        y_pred = nn.ReLU(y_pred)
        y_pred = self.linear2(y_pred)
        return y_pred


lstmInputSize = 4
lstmHLayers = 2
lstmHNodes = 128
lstmFCDIM = 32
lstmDropP = 0.3
lstmOutputDim = 2

learning_rate = 1e-4


model = LSTM(input_dim=lstmInputSize, hidden_dim=lstmHNodes, output_dim=lstmOutputDim, num_layers=lstmHLayers,
             drop_p=lstmDropP, linear_dim=lstmFCDIM)

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 1

def train(num_epochs):

