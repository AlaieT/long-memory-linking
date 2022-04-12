import torch
from torch import nn
import torch.nn.functional as F

# Create data

# -----------------------------------------------------------------------------
# -----------------------------Class defenition--------------------------------
# -----------------------------------------------------------------------------


class LSTM(nn.Module):
    def __init__(self, hidden_layers=16):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers

        self.lstmf = nn.LSTMCell(4, self.hidden_layers)
        self.lstmb = nn.LSTMCell(4, self.hidden_layers)

        self.lstmf2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.lstmb2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)

        self.lstmf3 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.lstmb3 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)

        self.linear = nn.Linear(self.hidden_layers*2, 4)

    def forward(self, y):
        outputs, num_samples = torch.tensor([]), y.size(0)
        h_tf = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_tf = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        h_tf2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_tf2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        h_tf3 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_tf3 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)

        h_tb = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_tb = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        h_tb2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_tb2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        h_tb3 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_tb3 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)

        f_y = y.split(1, dim=1)
        b_y = torch.flip(y, dims=[1]).split(1, dim=1)

        for i in range(len(f_y)):
            time_step_f = f_y[i]
            time_step_b = b_y[i]

            time_step_f = time_step_f.reshape(time_step_f.shape[0], time_step_f.shape[2])
            time_step_b = time_step_b.reshape(time_step_b.shape[0], time_step_b.shape[2])

            h_tf, c_tf = self.lstmf(time_step_f, (h_tf, c_tf))
            h_tf2, c_tf2 = self.lstmf2(h_tf, (h_tf2, c_tf2))
            h_tf3, c_tf3 = self.lstmf3(h_tf2, (h_tf3, c_tf3))

            h_tb, c_tb = self.lstmb(time_step_b, (h_tb, c_tb))
            h_tb2, c_tb2 = self.lstmb2(h_tb, (h_tb2, c_tb2))
            h_tb3, c_tb3 = self.lstmb3(h_tb2, (h_tb3, c_tb3))

        output = self.linear(torch.cat((h_tf3, h_tb3), dim=1))
        outputs = output.clone().unsqueeze(1)

        return outputs
