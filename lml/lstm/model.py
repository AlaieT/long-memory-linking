import torch
from torch import nn
import torch.nn.functional as F

# Create data


class LSTM(nn.Module):
    def __init__(self, hidden_layers=64):
        super(LSTM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_layers = hidden_layers

        self.lstmf = nn.LSTMCell(4, self.hidden_layers)
        self.lstmb = nn.LSTMCell(4, self.hidden_layers)

        self.linear1 = nn.Linear(self.hidden_layers*2, 1)
        self.linear2 = nn.Linear(self.hidden_layers*2, 1)
        self.linear3 = nn.Linear(self.hidden_layers*2, 1)
        self.linear4 = nn.Linear(self.hidden_layers*2, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, y):
        outputs, num_samples = torch.tensor([]), y.size(0)
        h_tf = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_tf = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)

        h_tb = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_tb = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)

        f_y = y.split(1, dim=1)
        b_y = torch.flip(y, dims=[1]).split(1, dim=1)

        for i in range(len(f_y)):
            temp_f = f_y[i]
            temp_b = b_y[i]

            temp_f = temp_f.reshape(temp_f.shape[0], temp_f.shape[2])
            temp_b = temp_b.reshape(temp_b.shape[0], temp_b.shape[2])

            h_tf, c_tf = self.lstmf(temp_f, (h_tf, c_tf))
            h_tb, c_tb = self.lstmb(temp_b, (h_tb, c_tb))

        h_tr = torch.cat((h_tf, h_tb), dim=1)

        _x = self.linear1(h_tr)
        _y = self.linear2(h_tr)
        _w = self.linear3(h_tr)
        _h = self.linear4(h_tr)

        output = torch.cat((_x, _y, _w, _h), dim=1)

        output = self.sigm(output)

        outputs = output.clone().unsqueeze(1)

        return outputs
