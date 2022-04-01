import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim

# Create data


N = 100  # number of samples
L = 1000  # length of each sample (number of values for each sine wave)
T = 20  # width of the wave
x = np.empty((N, L), np.float32)  # instantiate empty array
x[:] = np.arange(L) + np.random.randint(-4*T, 4*T, N).reshape(N, 1)
y = np.sin(x/1.0/T).astype(np.float32)

# Define lstm model


class LSTM(nn.Module):
    def __init__(self, hidden_layers=64):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(1, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 1)

    def forward(self, y, future_preds=0):
        outputs, num_samples = [], y.size(0)
        h_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)

        for time_step in y.split(1, dim=1):
            # N, 1
            h_t, c_t = self.lstm1(time_step, (h_t, c_t))  # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))  # new hidden and cell states
            output = self.linear(h_t2)  # output from the last FC layer
            outputs.append(output)

        if(future_preds != 0):
            outputs.clear()
            for i in range(future_preds):
                # this only generates future predictions if we pass in future_preds>0
                # mirrors the code above, using last output/prediction as input
                h_t, c_t = self.lstm1(output, (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                output = self.linear(h_t2)
                outputs.append(output)

        # transform list to tensor
        outputs = torch.cat(outputs, dim=1)
        return outputs

# train set


# y = (100, 1000)
train_input = torch.from_numpy(y[3:, :-1])  # (97, 999)
train_target = torch.from_numpy(y[3:, 1:])  # (97, 999)

# test set
test_input = torch.from_numpy(y[:1, :-1])  # (1, 999)
test_target = torch.from_numpy(y[:1, 1:])  # (1, 999)

model = LSTM()
criterion = nn.MSELoss()
optimiser = optim.LBFGS(model.parameters(), lr=0.08)


def training_loop(n_epochs, model, optimiser, loss_fn, train_input, train_target, test_input, test_target):
    # model.train()

    for i in range(n_epochs):
        model.train()

        def closure():
            optimiser.zero_grad()
            out = model(train_input)
            loss = loss_fn(out, train_target)
            loss.backward()
            return loss

        optimiser.step(closure)

        # print the loss
        out = model(train_input)
        loss_print = loss_fn(out, train_target)
        print(f"Step: {i+1}, Loss: {loss_print}")

        model.eval()

        with torch.no_grad():
            future_preds = 999
            pred = model(test_input, future_preds=future_preds)

            # use all pred samples, but only go to 999
            loss = loss_fn(pred, test_target)
            y = pred.detach().numpy()
            print(f"Valid loss: {loss}")

            # draw figures
            plt.figure(figsize=(12, 6))
            plt.title(f"Step {i+1}")
            plt.xlabel("x")
            plt.ylabel("y")
            n = test_input.shape[1]  # 999

            plt.plot(np.arange(n), test_input[0], 'r', linewidth=0.5)
            plt.plot(np.arange(n, n+future_preds), y[0], 'r'+":", linewidth=2.0)

            plt.savefig(f"predict{i+1}.png", dpi=200)
            plt.close()


training_loop(n_epochs=50,
              model=model,
              optimiser=optimiser,
              loss_fn=criterion,
              train_input=train_input,
              train_target=train_target,
              test_input=test_input,
              test_target=test_target)
