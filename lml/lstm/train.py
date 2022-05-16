from cmath import inf
import random
import numpy as np
import torch
from torch import nn, optim
import os
import matplotlib.pyplot as plt
from utils.get_data_from import get_data_from
from utils.plots import plot_heatmap, plot_track
from model import LSTM
from tqdm import tqdm
from sklearn.metrics import r2_score, pairwise_distances
import torch

# Train and valid data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

objects_train = get_data_from(path='./data/mot/MOT15/train', mod='train', device=device, folds=None)
print(f'\nTrain data lenght: {len(objects_train)}')

objects_valid = get_data_from(path='./data/mot/MOT15/valid', mod='valid', device=device, folds=None)
print(f'\nValid data lenght: {len(objects_valid)}')

# plot_track(objects_train[0][3])
# plot_heatmap(objects_train[0][3])

model_lstm = LSTM().to(device)
criterion = nn.MSELoss()
optimiser = optim.Adam(model_lstm.parameters(), lr=1e-3)


def training_loop(n_epochs, model, optimiser, loss_fn, train_data, test_data, save_path, metrick_path):

    loss_val = []
    loss_train = []

    min_val_loss = +inf

    for i in range(n_epochs):
        model.train()
        mean_loss = 0

        random.shuffle(train_data)

        print(f"\n------------------------------------> Epoch: {i+1}\n")

        print("\033[0;32m"+'############################### Train ###############################'+"\033[1;36m")
        with tqdm(total=len(train_data), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar_train:
            for data in train_data:
                for trajectory in data:
                    cur_input = trajectory[:, :-1, :]
                    cur_target = trajectory[:, -1:, :]

                    def clouser():
                        optimiser.zero_grad()
                        out = model(cur_input)
                        loss = loss_fn(out, cur_target)
                        loss.backward()
                        return loss

                    optimiser.step(clouser)

                    out = model(cur_input)
                    loss = loss_fn(out, cur_target)

                    mean_loss = mean_loss + loss.cpu().detach().numpy().tolist()

                pbar_train.update(1)
            pbar_train.close()

        loss_train.append(mean_loss)
        print("\033[0;0m"+f"\n\tTrain Loss: {mean_loss}\n")

        # print("\033[0;32m"+'############################### Valid ###############################'+"\033[1;36m")

        # model.eval()

        # # Cross train validation
        # with torch.no_grad():
        #     mean_loss = 0

        #     score_x_pred = []
        #     score_y_pred = []
        #     score_w_pred = []
        #     score_h_pred = []

        #     score_x_real = []
        #     score_y_real = []
        #     score_w_real = []
        #     score_h_real = []

        #     pair_pred = []

        #     with tqdm(total=len(test_data), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar_val:
        #         for data in test_data:
        #             for trajectory in data:
        #                 cur_test = trajectory[:, :-1, :]
        #                 cur_test_tar = trajectory[:, -1:, :]

        #                 pred = model(cur_test)
        #                 loss = loss_fn(pred, cur_test_tar)
        #                 outs = pred.cpu().detach().numpy().reshape((1, 4))

        #                 mean_loss = mean_loss + loss.cpu().detach().numpy().tolist()
        #                 real_xy = (trajectory.cpu())[:, -1:, :].reshape((1, 4))

        #                 score_x_pred.append(outs[0, 0])
        #                 score_y_pred.append(outs[0, 1])
        #                 score_w_pred.append(outs[0, 2])
        #                 score_h_pred.append(outs[0, 3])

        #                 score_x_real.append(real_xy[0, 0])
        #                 score_y_real.append(real_xy[0, 1])
        #                 score_w_real.append(real_xy[0, 2])
        #                 score_h_real.append(real_xy[0, 3])

        #                 pair_pred = np.append(pair_pred, pairwise_distances(outs, real_xy))

        #             pbar_val.update(1)
        #         pbar_val.close()

        #     loss_val.append(mean_loss)

        #     r2_x = r2_score(score_x_pred, score_x_real)
        #     r2_y = r2_score(score_y_pred, score_y_real)
        #     r2_w = r2_score(score_w_pred, score_w_real)
        #     r2_h = r2_score(score_h_pred, score_h_real)

        #     print("\033[0;0m" + f"\n\tVal loss: {mean_loss}\n")
        #     print("\033[0;0m" + f"\n\tX R2 Score: {r2_x}")
        #     print("\033[0;0m" + f"\n\tY R2 Score: {r2_y}")
        #     print("\033[0;0m" + f"\n\tW R2 Score: {r2_w}")
        #     print("\033[0;0m" + f"\n\tH R2 Score: {r2_h}")

        #     print("\033[0;0m" + f"\n\tPairwise distances MIN: {1-np.max(pair_pred)}")
        #     print("\033[0;0m" + f"\n\tPairwise distances MAX: {1-np.min(pair_pred)}")
        #     print("\033[0;0m" + f"\n\tPairwise distances MEAN: {1-np.mean(pair_pred)}")

        if(not os.path.exists(f'{save_path}')):
            os.mkdir(f'{save_path}')
        torch.save(model.state_dict(), f'{save_path}/last.pt')

        #     if(mean_loss < min_val_loss):
        #         min_val_loss = mean_loss
        #         torch.save(model.state_dict(), f'{save_path}/best.pt')

    '''
    Save metrick of current model
    '''
    if(not os.path.exists(f'{metrick_path}')):
        os.mkdir(f'{metrick_path}')

    fig, axs = plt.subplots(2, 1, figsize=(18, 18))
    fig.tight_layout()

    axs[0].plot(np.linspace(0, len(loss_train), len(loss_train)), loss_train)
    axs[0].set_title('Loss Train')
    axs[0].grid()

    axs[1].plot(np.linspace(0, len(loss_val), len(loss_val)), loss_val)
    axs[1].set_title('Loss Val')
    axs[1].grid()

    plt.savefig(f"{metrick_path}/metricks.png", dpi=250)
    plt.close()


training_loop(n_epochs=10, model=model_lstm, optimiser=optimiser, loss_fn=criterion, train_data=objects_train,
              test_data=objects_valid, save_path='./models/mot', metrick_path='./metricks/mot')
