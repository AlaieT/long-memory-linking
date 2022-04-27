import torch
from torch import nn
import torch.nn.functional as F

# Create data

# -----------------------------------------------------------------------------
# -----------------------------Class defenition--------------------------------
# -----------------------------------------------------------------------------


class LSTM(nn.Module):
    def __init__(self, hidden_layers=64):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers

        self.lstmf = nn.LSTMCell(4, self.hidden_layers)
        self.lstmb = nn.LSTMCell(4, self.hidden_layers)

        self.post = nn.Sequential(nn.utils.weight_norm(nn.Linear(self.hidden_layers*2, self.hidden_layers*4), dim=None),
                                  nn.Dropout(p=0.2),
                                  nn.utils.weight_norm(nn.Linear(self.hidden_layers*4, self.hidden_layers*2)),
                                  nn.utils.weight_norm(nn.Linear(self.hidden_layers*2,4)), nn.Sigmoid())


    def forward(self, y):
        outputs, num_samples = torch.tensor([]), y.size(0)
        h_tf = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32).cuda()
        c_tf = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32).cuda()

        h_tb = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32).cuda()
        c_tb = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32).cuda()

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

        out = self.post(h_tr)
        outputs = out.clone().unsqueeze(1)

        return outputs
