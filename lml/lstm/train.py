import random
import numpy as np
import torch
from torch import nn, optim
import os
import matplotlib.pyplot as plt
from utils.get_data_from import get_data_from
from model import LSTM
from tqdm import tqdm
from sklearn.metrics import r2_score


# Train and valid data
objects_train = get_data_from(path='./data/signate/train', mod='train', folds=None)
objects_valid = get_data_from(path='./data/signate/valid', mod='valid', folds=None)

model_lstm = LSTM().cuda()
criterion = nn.L1Loss()
optimiser = optim.Adam(model_lstm.parameters(), lr=1e-3)


def training_loop(n_epochs, model, optimiser, loss_fn, train_data, test_data, save_path, metrick_path):

    loss_val = []
    loss_train = []

    for i in range(n_epochs):
        model.train()
        mean_loss = 0

        random.shuffle(train_data)

        print(f"\n------------------------------------> Epoch: {i+1}\n")

        print("\033[0;32m"+'############################### Train ###############################'+"\033[1;36m")
        with tqdm(total=len(train_data), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar_train:
            for data in train_data:
                # get one sample
                if(data.shape[1] > 1):

                    cur_input = data[:, :-1, :]
                    cur_target = data[:, -1:, :]

                    def closure():
                        optimiser.zero_grad()
                        out = model(cur_input)
                        loss = loss_fn(out, cur_target)
                        loss.backward()
                        return loss

                    optimiser.step(closure)

                    out = model(cur_input)
                    loss = loss_fn(out, cur_target)

                    mean_loss = mean_loss + loss.cpu().detach().numpy().tolist()

                pbar_train.update(1)
            pbar_train.close()

        loss_train.append(mean_loss)
        print("\033[0;0m"+f"\n\tTrain Loss: {mean_loss}\n")

        print("\033[0;32m"+'############################### Valid ###############################'+"\033[1;36m")

        model.eval()

        # Cross train validation
        with torch.no_grad():
            mean_loss = 0

            score_x_pred = []
            score_y_pred = []
            score_w_pred = []
            score_h_pred = []

            score_x_real = []
            score_y_real = []
            score_w_real = []
            score_h_real = []

            with tqdm(total=len(test_data), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar_val:
                for data in test_data:
                    if(data.shape[1] > 1):

                        if(data.shape[1] >= 6):
                            cur_test = data[:, -6:-1, :]
                            cur_test_tar = data[:, -1:, :]
                        else:
                            cur_test = data[:, :-1, :]
                            cur_test_tar = data[:, -1:, :]

                        pred = model(cur_test)
                        loss = loss_fn(pred, cur_test_tar)
                        outs = pred.cpu().detach().numpy().reshape((4,))

                        mean_loss = mean_loss + loss.cpu().detach().numpy().tolist()
                        real_xy = (data.cpu())[:, -1:, :]

                        score_x_pred.append(outs[0])
                        score_y_pred.append(outs[1])
                        score_w_pred.append(outs[2])
                        score_h_pred.append(outs[3])

                        score_x_real.append(real_xy[0, 0, 0])
                        score_y_real.append(real_xy[0, 0, 1])
                        score_w_real.append(real_xy[0, 0, 2])
                        score_h_real.append(real_xy[0, 0, 3])

                    pbar_val.update(1)
                pbar_val.close()

            loss_val.append(mean_loss)

            r2_x = r2_score(score_x_pred,score_x_real)
            r2_y = r2_score(score_y_pred,score_y_real)
            r2_w = r2_score(score_w_pred,score_w_real)
            r2_h = r2_score(score_h_pred,score_h_real)

            print("\033[0;0m" + f"\n\tVal loss: {mean_loss}\n")
            print("\033[0;0m" + f"\n\tX R2 score: {r2_x}")
            print("\033[0;0m" + f"\n\tY R2 score: {r2_y}")
            print("\033[0;0m" + f"\n\tW R2 score: {r2_w}")
            print("\033[0;0m" + f"\n\tH R2 score: {r2_h}")

            if(not os.path.exists(f'{save_path}')):
                 os.mkdir(f'{save_path}')
            torch.save(model.state_dict(), f'{save_path}/last.pt')

    '''
    Save metrick of current model
    '''
    if(not os.path.exists(f'{metrick_path}/mot')):
        os.mkdir(f'{metrick_path}/mot')

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
