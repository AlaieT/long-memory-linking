import random
import numpy as np
import torch
from torch import nn, optim
import os
import matplotlib.pyplot as plt
# from utils.show_in_time import show_in_time
from utils.get_data_from import get_data_from
from model import LSTM
from tqdm import tqdm

# -----------------------------------------------------------------------------
# -----------------------------Train process-----------------------------------
# -----------------------------------------------------------------------------

# Train and valid data
objects_train, _ = get_data_from('./data/train', 'train')
objects_valid, _ = get_data_from('./data/valid', 'valid')

model_lstm = LSTM()
criterion = nn.BCELoss()
optimiser = optim.Adam(model_lstm.parameters(), lr=2e-3)


def training_loop(n_epochs, model, optimiser, loss_fn, train_data, test_data):

    loss_val = []
    loss_train = []
    dx_fr = []

    best_dx = 1

    for i in range(n_epochs):
        model.train()
        mean_loss = 0

        random.shuffle(train_data)

        print(f"\n------------------------------------> Epoch: {i+1}\n")

        print("\033[0;32m"+'############################### Train ###############################'+"\033[1;36m")
        with tqdm(total=len(train_data), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar_train:
            for k in range(0, len(train_data), 1):
                # get one sample
                samples = np.empty((1, len(train_data[k]), 4), dtype=np.float32)
                samples[0, :, :] = np.array(train_data[k], dtype=np.float32)

                if(samples.shape[1] > 1):

                    cur_input = samples[0, :-1, :]
                    cur_target = samples[0, -1, :]

                    cur_input = torch.tensor(cur_input.reshape((1, cur_input.shape[0], 4)))
                    cur_target = torch.tensor(cur_target.reshape((1, 1, 4)))

                    def closure():
                        optimiser.zero_grad()
                        out = model(cur_input)
                        loss = loss_fn(out, cur_target)
                        loss.backward()
                        return loss

                    optimiser.step(closure)

                    out = model(cur_input)
                    loss = loss_fn(out, cur_target)

                    mean_loss = mean_loss + loss.detach().numpy().tolist()

                pbar_train.update(1)
            pbar_train.close()

        loss_train.append(mean_loss)
        print("\033[0;0m"+f"\n\tTrain Loss: {mean_loss}\n")

        print("\033[0;32m"+'############################### Valid ###############################'+"\033[1;36m")

        model.eval()

        # Cross train validation
        with torch.no_grad():
            mean_loss = 0
            mean_dx = np.array([])
            mean_wh = np.array([])

            with tqdm(total=len(test_data), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar_val:
                for k in range(0, len(test_data), 1):
                    samples = np.empty((1, len(test_data[k]), 4), dtype=np.float32)
                    samples[0, :, :] = np.array(test_data[k], dtype=np.float32)

                    if(samples.shape[1] > 1):

                        if(samples.shape[1] >= 6):
                            cur_test = samples[0, -6:-1, :]
                            cur_test_tar = samples[0, -1, :]
                        else:
                            cur_test = samples[0, :-1, :]
                            cur_test_tar = samples[0, -1, :]

                        cur_test = torch.tensor(cur_test.reshape((1, cur_test.shape[0], 4)))
                        cur_test_tar = torch.tensor(cur_test_tar.reshape((1, 1, 4)))

                        pred = model(cur_test)
                        loss = loss_fn(pred, cur_test_tar)
                        outs = pred.detach().numpy().reshape((4,))

                        mean_loss = mean_loss + loss.detach().numpy().tolist()

                        real_xy = samples[0, -1, :]
                        out_xy = outs
                        dx = np.sqrt((real_xy[0]-out_xy[0])**2+(real_xy[1]-out_xy[1])**2)
                        wh = np.sqrt((real_xy[2]*real_xy[3]-out_xy[2]*real_xy[3])**2+(real_xy[3]-out_xy[3])**2)
                        mean_dx = np.append(mean_dx, dx)
                        mean_wh = np.append(mean_wh, wh)

                        # real = samples[0, :, :]

                        # Plot valid
                        # if(k == 10):
                        #     collect_out = [samples[0, 0, :].tolist()]
                        #     for p in range(1, samples.shape[1], 1):
                        #         cur_test = samples[0, :p, :]
                        #         cur_test = torch.tensor(cur_test.reshape((1, cur_test.shape[0], 2)))

                        #         pred = model(cur_test)
                        #         outs = pred.detach().numpy().reshape((2,))

                        #         collect_out.append(outs)

                        #     show_in_time(collect_out,  real, f'step_{i+1}_{k}')

                    pbar_val.update(1)
                pbar_val.close()

            dx_fr.append(np.mean(mean_dx))
            loss_val.append(mean_loss)

            print("\033[0;0m" + f"\n\tVal loss: {mean_loss}\n")
            print(f'\tMin Dx: {np.min(mean_dx)} Min WH: {np.min(mean_wh)}\n')
            print(f'\tMax Dx: {np.max(mean_dx)} Max WH: {np.max(mean_wh)}\n')
            print(f'\tMean Dx: {np.mean(mean_dx)} Mean WH: {np.mean(mean_wh)}')

            if(best_dx > np.mean(mean_dx)):
                best_dx = np.mean(mean_dx)
                if(not os.path.exists('./models')):
                    os.mkdir('./models')
                torch.save(model_lstm.state_dict(), './models/best.pt')

    return loss_train, loss_val, dx_fr


loss_train, loss_val, dx_fr = training_loop(n_epochs=25, model=model_lstm, optimiser=optimiser,
                                            loss_fn=criterion, train_data=objects_train, test_data=objects_valid)

# -----------------------------------------------------------------------------
# ------------------------------Save metrick data------------------------------
# -----------------------------------------------------------------------------

# Create a visualization

if(not os.path.exists('./metricks')):
    os.mkdir('./metricks')

fig, axs = plt.subplots(3, 1, figsize=(18, 18))
fig.tight_layout()

axs[0].plot(np.linspace(0, len(loss_train), len(loss_train)), loss_train)
axs[0].set_title('Loss Train')
axs[0].grid()

axs[1].plot(np.linspace(0, len(loss_val), len(loss_val)), loss_val)
axs[1].set_title('Loss Val')
axs[1].grid()

axs[2].plot(np.linspace(0, len(dx_fr), len(dx_fr)), dx_fr)
axs[2].set_title('Dx')
axs[2].grid()

plt.savefig(f"./metricks/metricks.png", dpi=250)
plt.close()


if(not os.path.exists('./models')):
    os.mkdir('./models')

torch.save(model_lstm.state_dict(), './models/last.pt')
