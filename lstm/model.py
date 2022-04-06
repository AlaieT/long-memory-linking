import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import re
import os
import json

# Create data

trajectory = []
trajectory_count = 50
trajectory_accurasy = 100


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


annotation_files = []  # files for traning and vlidation

if(os.path.exists('./validation/true')):
    annotation_files = os.listdir('./validation/true')
    annotation_files.sort(key=natural_keys)

    in_frame_cars = []
    in_frame_pedestrians = []

    with open(f'./validation/true/{annotation_files[0]}') as f:
        true_json = json.load(f)['sequence']
        for frame in true_json:
            in_frame_cars += frame['Car']
            in_frame_pedestrians += frame['Pedestrians']

        def get_my_key(obj):
            return obj['id']

        in_frame_cars.sort(key=get_my_key)
        in_frame_pedestrians.sort(key=get_my_key)

        


class LSTM(nn.Module):
    def __init__(self, hidden_layers=16):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(3, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 3)

    def forward(self, input_frames, future_preds=False):
        outputs_frames = []
        for frame in input_frames.split(1, dim=0):
            # some comment
            frame = frame.squeeze(0)

            # for data_i in range(frame.shape[0]):
            #     if(frame[data_i][0] == -1):
            #         frame = frame[:data_i-1, :]
            #         break

            outputs, num_samples = [], frame.size(0)
            h_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
            c_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
            h_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
            c_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)

            # N, 1
            h_t, c_t = self.lstm1(frame, (h_t, c_t))  # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))  # new hidden and cell states
            output = self.linear(h_t2)  # output from the last FC layer
            # only when traning
            if(not future_preds):
                outputs.append(output.tolist())

            if(future_preds):
                # this only generates future predictions if we pass in future_preds>0
                # mirrors the code above, using last output/prediction as input
                h_t, c_t = self.lstm1(output, (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                output = self.linear(h_t2)
                outputs.append(output.tolist())

            outputs_frames += outputs
        # transform list to tensor
        outputs_frames = torch.tensor(outputs_frames, requires_grad=True)
        return outputs_frames


# # train set
# train_input = torch.tensor(frame_data[:120][:-1])  # (97, 999)
# train_target = torch.tensor(frame_data[:120][1:])  # (97, 999)

# # test se
# test_input = torch.tensor(frame_data[120:][:-1])  # (1, 999)
# test_target = torch.tensor(frame_data[120:][1:])  # (1, 999)


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
            pred = model(test_input, future_preds=True)

            loss = loss_fn(pred, test_target)
            y = pred.detach().numpy()
            print(f"Valid loss: {loss}")

            plt.figure(figsize=(12, 6))
            plt.title(f"Step {i+1}")
            plt.xlabel("x")
            plt.ylabel("y")

            for frame in range(y.shape[0]):
                # draw figures
                for object in range(y.shape[1]):
                    plt.plot(train_input[:, object, 1].tolist() + y[frame, object, 1],
                             train_input[:, object, 2].tolist() + y[frame, object, 1], linewidth=0.5)

            plt.savefig(f"predict{i+1}.png", dpi=200)
            plt.close()


# training_loop(n_epochs=50,
#               model=model,
#               optimiser=optimiser,
#               loss_fn=criterion,
#               train_input=train_input,
#               train_target=train_target,
#               test_input=test_input,
#               test_target=test_target)
